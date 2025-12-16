#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda/std/atomic>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>

#include "helper_headers/json.hpp"

#include "helper_headers/csrmat.hpp"
#include "helper_headers/fassert.cuh"
#include "helper_headers/gpuErrChk.h"

#include "macro.h"
#include "export.cuh"

#include "../work_t.cuh"

// #define WORK_DYNAMIC 1
#define WORK_DYNAMIC 0

typedef uint64_t vertex_t;
typedef uint32_t value_t;
// typedef float value_t;

struct __align__(8) app_data {
    memalloc_dptr_t edge_dptr_memalloc;
    memalloc_dptr_t value_dptr_memalloc;
    vertex_t parent_start_vid;
    uint64_t num_total_children;

    uint64_t value_cid;
    uint32_t edge_start_align_add;
    uint32_t value_start_align_add;

    uint32_t level;
};

template <typename T>
struct __align__(4) work_padded_atomic {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[4];
    };
};

class __align__(128) work
    : public work_base {
    public:
    uint32_t *level;

    // threshold
    uint32_t update;

    vertex_t *vertex_list;
    uint64_t vertex_count;
    uint64_t edge_list_len;

    cuda::atomic<value_t, cuda::thread_scope_device> *cost_list;
    cuda::atomic<value_t, cuda::thread_scope_device> *new_cost_list;

    cuda::atomic<uint64_t, cuda::thread_scope_device> vid_counter;

    uint64_t edge_list_nvme_start_byte;
    uint64_t value_list_nvme_start_byte;

    // From json
    uint32_t nvme_dev_id;
    uint64_t edge_list_nvme_start_GiB;
    uint64_t value_list_nvme_start_GiB;
    vertex_t root;
    uint32_t max_level;

    __device__ void work_init_d() {
        printf("%s: sizeof(app_data): %lu\n", __func__, sizeof(struct app_data));
    }
    __device__ void work_fini_d() {
    }

    __host__ void work_init_h(const uint32_t num_controllers, const nlohmann::json &in_js,
                              std::vector<struct bam_array_in> &bam_array_in_vec) {
        static_assert(std::is_standard_layout_v<work> == true);
        static_assert(std::is_trivially_copyable_v<work> == true);

        std::remove_pointer_t<decltype(this)> wd;
        wd.wb_init_h();

        std::string graph_filename;
        try {
            graph_filename = in_js.at("graph_filename");
            wd.nvme_dev_id = in_js.at("nvme_dev_id");
            wd.edge_list_nvme_start_GiB = in_js.at("edge_list_nvme_start_GiB");
            wd.value_list_nvme_start_GiB = in_js.at("value_list_nvme_start_GiB");
            wd.root = in_js.at("root");
            wd.max_level = in_js.at("max_level");
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            force_assert(false);
        }
        wd.edge_list_nvme_start_byte = wd.edge_list_nvme_start_GiB * 1024 * 1024 * 1024;
        wd.value_list_nvme_start_byte = wd.value_list_nvme_start_GiB * 1024 * 1024 * 1024;

        // Read CSR matrix
        class Csrmat<vertex_t, value_t> mat(graph_filename);
        mat.read_mat();
        // force_assert(mat.symmetric == true); // Directed graph?
        force_assert(mat.num_rows == mat.num_cols);
        force_assert_printf(mat.pattern == false, "Graph must have weights!\n");
        wd.vertex_count = mat.num_rows;
        wd.edge_list_len = mat.edge_list_len;
        cudaErrChk(cudaMalloc(&(wd.vertex_list), sizeof(vertex_t) * (mat.num_rows + 1)));
        cudaErrChk(cudaMemcpy(wd.vertex_list, mat.vertex_list, sizeof(vertex_t) * (mat.num_rows + 1), cudaMemcpyHostToDevice));

        cudaErrChk(cudaMalloc(&wd.level, mat.num_rows * sizeof(*(wd.level))));
        cudaErrChk(cudaMemset(wd.level, 0xFF, mat.num_rows * sizeof(*(wd.level))));
        force_assert(wd.root < mat.num_rows);

        // bam
        for (uint32_t i = 0; i < num_controllers; ++i) {
            struct bam_array_in edge_range{};
            edge_range.nvme_ctrl_id = i;
            edge_range.bam_array_nvme_start_byte = wd.edge_list_nvme_start_byte;
            edge_range.bam_array_size_bytes = wd.edge_list_len * sizeof(vertex_t);
            edge_range.bam_array_name = "edge";
            bam_array_in_vec.push_back(edge_range);

            struct bam_array_in value_range{};
            value_range.nvme_ctrl_id = i;
            value_range.bam_array_nvme_start_byte = wd.value_list_nvme_start_byte;
            value_range.bam_array_size_bytes = wd.edge_list_len * sizeof(value_t);
            value_range.bam_array_name = "weight";
            bam_array_in_vec.push_back(value_range);
        }

        // Cost list
        cudaErrChk(cudaMalloc(&wd.cost_list, sizeof(*(wd.cost_list)) * wd.vertex_count));
        cudaErrChk(cudaMalloc(&wd.new_cost_list, sizeof(*(wd.new_cost_list)) * wd.vertex_count));
        value_t *cost_list_h = new value_t[wd.vertex_count];
        for (uint64_t i = 0; i < wd.vertex_count; ++i) {
            cost_list_h[i] = std::numeric_limits<value_t>::max();
        }
        cudaErrChk(cudaMemcpy(wd.cost_list, cost_list_h, sizeof(value_t) * wd.vertex_count, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(wd.new_cost_list, cost_list_h, sizeof(value_t) * wd.vertex_count, cudaMemcpyHostToDevice));
        delete[] cost_list_h;

        memcpy(this, &wd, sizeof(*this));
    }
    __host__ void work_fini_h() {
        std::remove_pointer_t<decltype(this)> wd;
        memcpy(reinterpret_cast<char *>(&wd), this, sizeof(*this));

        cudaErrChk(cudaFree(wd.vertex_list));
        cudaErrChk(cudaFree(wd.level));
        cudaErrChk(cudaFree(wd.cost_list));
        cudaErrChk(cudaFree(wd.new_cost_list));

        wd.wb_fini_h();
    }
};

struct __align__(32) work_w
    : public work_base_w<class work> {
    public:
    __host__ void
    work_w_init_h(const uint32_t num_controllers, nlohmann::json in_js) {
        wd_h.work_init_h(num_controllers, in_js, bam_array_in_vec);
        cudaErrChk(cudaMalloc(&(wd_dp), sizeof(*wd_dp)));
        cudaErrChk(cudaMemcpy(wd_dp, &wd_h, sizeof(*wd_dp), cudaMemcpyHostToDevice));
    }
    __host__ void
    work_w_fini_h() {
        cudaErrChk(cudaFree(wd_dp));
        wd_h.work_fini_h();
    }
    __host__ void
    work_print_stats(const uint32_t gpu_clock_mhz, nlohmann::json &out_js);
};

__host__ void
work_w::work_print_stats(const uint32_t gpu_clock_mhz, nlohmann::json &out_js) {
    class work wd_temp_h;
    cudaErrChk(cudaMemcpy(&wd_temp_h, wd_dp, sizeof(class work), cudaMemcpyDeviceToHost));

    // Export the level array (double-check)
    uint64_t num_vertices = wd_temp_h.vertex_count;
    uint32_t *level_h = new uint32_t[num_vertices]{};
    cudaErrChk(cudaMemcpy(level_h, wd_temp_h.level, num_vertices * sizeof(*level_h), cudaMemcpyDeviceToHost));
    uint64_t sum = 0;
    std::vector<size_t> level_sum(0);
    for (uint64_t i = 0; i < num_vertices; ++i) {
        if (level_h[i] != UINT32_MAX) {
            sum++;
            if (level_h[i] + 1 > level_sum.size())
                level_sum.resize(level_h[i] + 1);
            level_sum[level_h[i]]++;
        }
    }
    for (size_t i = 0; i < level_sum.size(); ++i) {
        printf("%s: level: %lu, level_sum[]: %lu\n", __func__, i, level_sum[i]);
    }

    // Write to file
    FILE *fp_level = fopen("result-" C_WORK_NAME "-level-array.data", "w");
    fwrite(level_h, sizeof(*level_h), num_vertices, fp_level);
    fclose(fp_level);

    delete[] level_h;
    printf("%s: n visited vertices: %lu, original vertex count: %lu\n", __func__, sum, wd_temp_h.vertex_count);
    out_js.emplace("work_n_visited_vertices", sum);

    // Export the cost array
    std::vector<value_t> cost_list_h(num_vertices);
    cudaErrChk(cudaMemcpy(cost_list_h.data(), wd_temp_h.cost_list, sizeof(cost_list_h[0]) * num_vertices, cudaMemcpyDeviceToHost));
    FILE *fp_cost = fopen("result-" C_WORK_NAME "-cost-array.data", "w");
    fwrite(cost_list_h.data(), sizeof(cost_list_h[0]), num_vertices, fp_cost);
    fclose(fp_cost);
}

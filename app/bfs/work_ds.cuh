#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <type_traits>

#include <cuda/atomic>
#include <cuda.h>

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
typedef void value_t;

struct __align__(8) app_data {
    memalloc_dptr_t memalloc_dptr;
    vertex_t parent_start_vid;
    uint32_t num_total_children;
    uint32_t child_start_align_add;
    uint32_t level;
};

template <typename T>
struct __align__(4) work_padded_atomic {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[4];
    };
};

class __align__(32) work
    : public work_base {
    public:
    uint32_t *level;
    uint32_t update;

    vertex_t *vertex_list;
    uint64_t vertex_count;

    vertex_t *edge_list;
    uint64_t edge_list_len;

    cuda::atomic<uint64_t, cuda::thread_scope_device> vid_counter;

    uint64_t nvme_edge_start_byte;

    // From json
    uint32_t nvme_dev_id;
    uint64_t nvme_edge_start_GiB;
    vertex_t root;
    uint32_t max_level;

    __device__ void work_init_d() {
        wb_init_d();
        level[root] = 0;
        printf("%s: root: %lu, sizeof(app_data): %lu, set level to zero.\n", __func__, root, sizeof(struct app_data));

        vid_counter.store(0);
    }
    __device__ void work_fini_d() {
        wb_fini_d();
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
            wd.nvme_edge_start_GiB = in_js.at("nvme_edge_start_GiB");
            wd.root = in_js.at("root");
            wd.max_level = in_js.at("max_level");
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            force_assert(false);
        }
        wd.nvme_edge_start_byte = wd.nvme_edge_start_GiB * 1024 * 1024 * 1024;

        // Read CSR matrix
        class Csrmat<vertex_t, value_t> mat(graph_filename);
        mat.read_mat();
        // force_assert(mat.symmetric == true); // Directed graph?
        force_assert(mat.num_rows == mat.num_cols);
        wd.vertex_count = mat.num_rows;
        wd.edge_list_len = mat.edge_list_len;
        printf("%s: graph_filename: %s\n", __func__, graph_filename.c_str());
        cudaErrChk(cudaMalloc(&(wd.vertex_list), sizeof(vertex_t) * (mat.num_rows + 1)));
        cudaErrChk(cudaMemcpy(wd.vertex_list, mat.vertex_list, sizeof(vertex_t) * (mat.num_rows + 1), cudaMemcpyHostToDevice));

        cudaErrChk(cudaMalloc(&(wd.level), mat.num_rows * sizeof(*(wd.level))));
        cudaErrChk(cudaMemset(wd.level, 0xFF, mat.num_rows * sizeof(*(wd.level))));
        force_assert(wd.root < mat.num_rows);

        for (uint32_t i = 0; i < num_controllers; ++i) {
            struct bam_array_in temp{};
            temp.nvme_ctrl_id = i;
            temp.bam_array_nvme_start_byte = wd.nvme_edge_start_byte;
            temp.bam_array_size_bytes = wd.edge_list_len * sizeof(vertex_t);
            temp.bam_array_name = "edge";
            bam_array_in_vec.push_back(temp);
        }

        memcpy(this, &wd, sizeof(*this));
    }
    __host__ void work_fini_h() {
        std::remove_pointer_t<decltype(this)> wd;
        memcpy(reinterpret_cast<char *>(&wd), this, sizeof(*this));

        cudaErrChk(cudaFree(wd.level));
        cudaErrChk(cudaFree(wd.vertex_list));

        wd.wb_fini_h();
    }

    __forceinline__ __device__ int32_t
    build_iocb(class gio *gp, class iocb *cb, uint64_t vid, uint32_t local_level);
};

class __align__(32) work_w
    : public work_base_w<class work> {
    public:
    __host__ void
    work_w_init_h(const uint32_t num_controllers, nlohmann::json in_js) {
        wb_w_init_h();
        wd_h.work_init_h(num_controllers, in_js, bam_array_in_vec);
        cudaErrChk(cudaMalloc(&(wd_dp), sizeof(*wd_dp)));
        cudaErrChk(cudaMemcpy(wd_dp, &wd_h, sizeof(*wd_dp), cudaMemcpyHostToDevice));
    }
    __host__ void
    work_w_fini_h() {
        cudaErrChk(cudaFree(wd_dp));
        wd_h.work_fini_h();
        wb_w_fini_h();
    }

    __host__ void
    work_print_stats(const uint32_t gpu_clock_mhz, nlohmann::json &out_js);
};

__host__ void
work_w::work_print_stats(const uint32_t gpu_clock_mhz, nlohmann::json &out_js) {
    class work wd_temp_h;
    cudaErrChk(cudaMemcpy(&wd_temp_h, wd_dp, sizeof(*wd_dp), cudaMemcpyDeviceToHost));

    // Count the visited array (double-check)
    uint64_t num_vertices = wd_temp_h.vertex_count;
    std::vector<uint32_t> level_h(num_vertices);
    cudaErrChk(cudaMemcpy(level_h.data(), wd_temp_h.level, num_vertices * sizeof(level_h[0]), cudaMemcpyDeviceToHost));
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
    std::string result_filename = "result-" C_WORK_NAME "-level-array.data";
    FILE *fp = fopen(result_filename.c_str(), "w");
    fwrite(level_h.data(), sizeof(level_h[0]), num_vertices, fp);
    fclose(fp);

    printf("%s: n visited vertices: %lu, original vertex count: %lu\n", __func__, sum, wd_temp_h.vertex_count);
    out_js.emplace("work_n_visited_vertices", sum);

}

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
    uint32_t child_start_align_add;
    uint32_t num_total_children;
};

class __align__(32) work
    : public work_base {
    public:
    uint32_t *level;

    cuda::atomic<uint64_t, cuda::thread_scope_device> visited_count;
    cuda::atomic<uint64_t, cuda::thread_scope_device> vid_counter;
    uint32_t changed;

    vertex_t *vertex_list;
    uint64_t vertex_count;
    vertex_t *edge_list;
    uint64_t edge_list_len;

    vertex_t *comp;
    alignas(16) bool *curr_visit;
    alignas(16) bool *next_visit;

    uint64_t nvme_edge_start_byte;

    // From json
    uint32_t nvme_dev_id;
    uint64_t nvme_edge_start_GiB;
    uint32_t max_level;

    __device__ void work_init_d() {
        wb_init_d();
        visited_count.store(0);
    }
    __device__ void work_fini_d() {
        printf("%s: visited_count: %lu\n", __func__, visited_count.load());
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
        cudaErrChk(cudaMalloc(&(wd.vertex_list), sizeof(vertex_t) * (mat.num_rows + 1)));
        cudaErrChk(cudaMemcpy(wd.vertex_list, mat.vertex_list, sizeof(vertex_t) * (mat.num_rows + 1), cudaMemcpyHostToDevice));

        // cudaErrChk(cudaMalloc(&wd_h_init.level, mat.num_rows * sizeof(*(wd_h_init.level))));
        // cudaErrChk(cudaMemset(wd_h_init.level, 0xFF, mat.num_rows * sizeof(*(wd_h_init.level))));
        // cudaErrChk(cudaMemset(wd_h_init.level, 0x00, mat.num_rows * sizeof(*(wd_h_init.level))));

        // CC data structures
        uint64_t vertex_count = mat.num_rows;
        vertex_t *comp_h = new vertex_t[vertex_count];
        for (uint64_t i = 0; i < vertex_count; ++i) {
            comp_h[i] = i;
        }
        cudaErrChk(cudaMalloc(&wd.comp, vertex_count * sizeof(*comp_h)));
        cudaErrChk(cudaMemcpy(wd.comp, comp_h, vertex_count * sizeof(*comp_h), cudaMemcpyHostToDevice));
        delete[] comp_h;
        cudaErrChk(cudaMalloc(&wd.curr_visit, vertex_count * sizeof(*wd.curr_visit)));
        cudaErrChk(cudaMemset(wd.curr_visit, 0xFF, vertex_count * sizeof(*wd.curr_visit)));
        cudaErrChk(cudaMalloc(&wd.next_visit, vertex_count * sizeof(*wd.curr_visit)));
        cudaErrChk(cudaMemset(wd.next_visit, 0x00, vertex_count * sizeof(*wd.curr_visit)));

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

        cudaErrChk(cudaFree(wd.next_visit));
        cudaErrChk(cudaFree(wd.curr_visit));
        cudaErrChk(cudaFree(wd.comp));
        // cudaErrChk(cudaFree(wd.level));
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

    // Count the number of components
    uint64_t num_vertices = wd_temp_h.vertex_count;
    std::vector<vertex_t> comp_h(num_vertices);
    cudaErrChk(cudaMemcpy(comp_h.data(), wd_temp_h.comp, num_vertices * sizeof(comp_h[0]), cudaMemcpyDeviceToHost));

    uint64_t num_components = 0;
    std::vector<bool> comp_check(num_vertices, false);
    for (uint64_t i = 0; i < num_vertices; ++i) {
        if (comp_check[comp_h[i]] == false) {
            comp_check[comp_h[i]] = true;
            num_components++;
        }
    }

    // Write to file
    std::string result_filename = "result-" C_WORK_NAME "-comp-array.data";
    FILE *fp = fopen(result_filename.c_str(), "w");
    fwrite(comp_h.data(), sizeof(comp_h[0]), num_vertices, fp);
    fclose(fp);

    printf("%s: number of components: %lu, graph vertex count: %lu\n", __func__, num_components, wd_temp_h.vertex_count);
    out_js.emplace("work_n_components", num_components);
}

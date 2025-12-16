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
typedef float aux_t;

struct __align__(8) app_data {
    memalloc_dptr_t memalloc_dptr;
    vertex_t parent_start_vid;
    uint32_t num_total_children;
    uint32_t child_start_align_add;
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
    vertex_t *vertex_list;
    uint64_t vertex_count;
    uint64_t edge_list_len;

    bool *label;
    aux_t *delta;
    aux_t *residual;
    aux_t *value;

    cuda::atomic<uint64_t, cuda::thread_scope_device> vid_counter;

    uint64_t nvme_edge_start_byte;

    // From json
    uint32_t nvme_dev_id;
    uint64_t nvme_edge_start_GiB;
    vertex_t root;
    uint32_t max_level;
    uint32_t tile_method;
    uint64_t tile_size;
    aux_t alpha;
    aux_t tolerance;

    __device__ void work_init_d() {
        wb_init_d();

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
            wd.max_level = in_js.at("max_level");
            wd.alpha = in_js.at("alpha");
            wd.tolerance = in_js.at("tolerance");
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

        for (uint32_t i = 0; i < num_controllers; ++i) {
            struct bam_array_in temp{};
            temp.nvme_ctrl_id = i;
            temp.bam_array_nvme_start_byte = wd.nvme_edge_start_byte;
            temp.bam_array_size_bytes = wd.edge_list_len * sizeof(vertex_t);
            temp.bam_array_name = "edge";
            bam_array_in_vec.push_back(temp);
        }

        // Aux data
        bool *label_h = new bool[wd.vertex_count];
        aux_t *delta_h = new aux_t[wd.vertex_count];
        aux_t *residual_h = new aux_t[wd.vertex_count];
        aux_t *value_h = new aux_t[wd.vertex_count];
        for (uint64_t i = 0; i < wd.vertex_count; ++i) {
            value_h[i] = 1.0f - wd.alpha;
            uint64_t num_edges = mat.vertex_list[i + 1] - mat.vertex_list[i];
            if (num_edges > 0) {
                delta_h[i] = (1.0f - wd.alpha) * wd.alpha / (mat.vertex_list[i + 1] - mat.vertex_list[i]);
            } else {
                delta_h[i] = 0.0f;
            }
            residual_h[i] = 0.0f;
            label_h[i] = true;
        }
        cudaErrChk(cudaMalloc(&(wd.label), sizeof(bool) * wd.vertex_count));
        cudaErrChk(cudaMalloc(&(wd.delta), sizeof(aux_t) * wd.vertex_count));
        cudaErrChk(cudaMalloc(&(wd.residual), sizeof(aux_t) * wd.vertex_count));
        cudaErrChk(cudaMalloc(&(wd.value), sizeof(aux_t) * wd.vertex_count));
        cudaErrChk(cudaMemcpy(wd.label, label_h, sizeof(bool) * wd.vertex_count, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(wd.delta, delta_h, sizeof(aux_t) * wd.vertex_count, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(wd.residual, residual_h, sizeof(aux_t) * wd.vertex_count, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(wd.value, value_h, sizeof(aux_t) * wd.vertex_count, cudaMemcpyHostToDevice));
        delete[] label_h;
        delete[] delta_h;
        delete[] residual_h;
        delete[] value_h;

        memcpy(this, &wd, sizeof(*this));
    }
    __host__ void work_fini_h() {
        std::remove_pointer_t<decltype(this)> wd;
        memcpy(reinterpret_cast<char *>(&wd), this, sizeof(*this));

        cudaErrChk(cudaFree(wd.value));
        cudaErrChk(cudaFree(wd.residual));
        cudaErrChk(cudaFree(wd.delta));
        cudaErrChk(cudaFree(wd.label));

        cudaErrChk(cudaFree(wd.vertex_list));

        wd.wb_fini_h();
    }

    __forceinline__ __device__ int32_t
    build_iocb(class gio *gp, struct iocb *cb, uint64_t vid);
    __forceinline__ __device__ void
    process_residual(class gio *gp, struct app_data *data);
    __forceinline__ __device__ void
    process_residual2(class gio *gp, struct app_data *data);

    __forceinline__ __device__ void
    update_func_ext(class gio *gp, vertex_t vid, uint32_t *changed);
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

    // Write to file
    std::vector<aux_t> value_h(wd_temp_h.vertex_count);
    cudaErrChk(cudaMemcpy(value_h.data(), wd_temp_h.value, sizeof(aux_t) * wd_temp_h.vertex_count, cudaMemcpyDeviceToHost));
    FILE *fp_pagerank = fopen("result-" C_WORK_NAME "-pagerank-array.data", "w");
    fwrite(value_h.data(), sizeof(value_h[0]), wd_temp_h.vertex_count, fp_pagerank);
    fclose(fp_pagerank);
}

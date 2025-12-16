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

#include "helper_headers/check.hpp"
#include "helper_headers/csrmat.hpp"
#include "helper_headers/fassert.cuh"
#include "helper_headers/gpuErrChk.h"

#include "macro.h"
#include "export.cuh"
#include "../work_t.cuh"

// #define WORK_DYNAMIC 1
#define WORK_DYNAMIC 0

// typedef double value_t;
typedef float value_t;

struct g_io_ret {
    uint32_t cid;
    void *dptr;
};

struct __align__(32) d_store {
    void *memalloc_dptr;
    uint64_t start_row;
    uint32_t align_add;
};

class __align__(32) work
    : public work_base {
    public:
    cuda::atomic<uint64_t, cuda::thread_scope_device> global_counter;

    // Matrix
    uint64_t num_rows;
    uint64_t num_cols;
    uint64_t matrix_nvme_start_byte;

    // Vectors
    value_t *vecX; // Input
    value_t *vecY; // Result

    uint64_t iteration_count;

    // From json
    uint32_t matrix_nvme_dev_id;
    uint64_t matrix_nvme_start_GiB;
    uint64_t num_rows_per_warp_per_loop;
    uint32_t issue_pending_threshold;
    uint32_t wait_pending_threshold;

    __device__ void work_init_d() {
        global_counter.store(0);
    }
    __device__ void work_fini_d() {
    }

    __host__ void work_init_h(const uint32_t num_controllers, const nlohmann::json &in_js,
                              std::vector<struct bam_array_in> &bam_array_in_vec) {
        static_assert(std::is_standard_layout_v<work> == true);
        static_assert(std::is_trivially_copyable_v<work> == true);

        std::remove_pointer_t<decltype(this)> wd;
        wd.wb_init_h();

        std::string matrix_filename;
        try {
            matrix_filename = in_js.at("matrix_filename");
            wd.matrix_nvme_dev_id = in_js.at("nvme_dev_id");
            wd.matrix_nvme_start_GiB = in_js.at("matrix_nvme_start_GiB");
            wd.num_rows_per_warp_per_loop = in_js.at("num_rows_per_warp_per_loop");
        } catch (const std::exception &e) {
            std::cerr << __func__ << ": " << e.what() << '\n';
            force_assert(false);
        }
        wd.matrix_nvme_start_byte = wd.matrix_nvme_start_GiB * 1024UL * 1024UL * 1024UL;

        // Read metadata
        std::ifstream f(matrix_filename + ".json");
        nlohmann::json mat_json = nlohmann::json::parse(f);
        f.close();

        wd.num_rows = mat_json.at("num_rows");
        wd.num_cols = mat_json.at("num_cols");

        // Check type
        g_typecheck(typeid(value_t), mat_json.at("value_type"));

        // Read input vector X file
        std::string vecX_filename = matrix_filename + ".vectorX";
        FILE *vecX_file = fopen(vecX_filename.c_str(), "r");
        if (!vecX_file) {
            fprintf(stderr, "Vector file open error.\n");
            exit(EXIT_FAILURE);
        }
        value_t *vecX_h = (value_t *)malloc(sizeof(value_t) * wd.num_cols);
        size_t ret = fread(vecX_h, sizeof(value_t), wd.num_cols, vecX_file);
        if (ret != wd.num_cols) {
            fprintf(stderr, "Vector file read error.\n");
            exit(EXIT_FAILURE);
        }
        fclose(vecX_file);
        cudaErrChk(cudaMalloc(&(wd.vecX), sizeof(value_t) * wd.num_cols));
        cudaErrChk(cudaMemcpy(wd.vecX, vecX_h, sizeof(value_t) * wd.num_cols, cudaMemcpyHostToDevice));
        free(vecX_h);

        // Result vecY
        cudaErrChk(cudaMalloc(&(wd.vecY), sizeof(value_t) * wd.num_rows));
        cudaErrChk(cudaMemset(wd.vecY, 0, sizeof(value_t) * wd.num_rows));

        memcpy(this, &wd, sizeof(*this));
    }
    __host__ void work_fini_h() {
        std::remove_pointer_t<decltype(this)> wd;
        memcpy(reinterpret_cast<char *>(&wd), this, sizeof(*this));

        cudaErrChk(cudaFree(wd.vecY));
        cudaErrChk(cudaFree(wd.vecX));

        wd.wb_fini_h();
    }
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

    // Write result to file
    uint64_t M = wd_temp_h.num_rows;
    std::vector<value_t> vecY_h(M);
    cudaErrChk(cudaMemcpy(vecY_h.data(), wd_temp_h.vecY, sizeof(value_t) * M, cudaMemcpyDeviceToHost));
    std::string vecY_filename = "result-" C_WORK_NAME ".vectorY";
    remove(vecY_filename.c_str());
    FILE *fp = fopen(vecY_filename.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Result vector open error.\n");
        exit(EXIT_FAILURE);
    }
    size_t ret = fwrite(vecY_h.data(), sizeof(value_t), M, fp);
    if (ret != M) {
        fprintf(stderr, "Result vector write error.\n");
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    out_js["iteration_count"] = wd_temp_h.iteration_count;
    size_t mat_size_bytes = wd_temp_h.num_rows * wd_temp_h.num_cols * sizeof(value_t);
    out_js.emplace("matrix_size_bytes", mat_size_bytes);
}

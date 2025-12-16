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

// #define WORK_DYNAMIC_POINT 1
#define WORK_DYNAMIC_POINT 0
#define WORK_MAX_MULT 16

#define WORK_HOST_MEM 1
// #define WORK_HOST_MEM 0

typedef float value_t;
constexpr value_t VALUE_T_MAX = std::numeric_limits<value_t>::max();

typedef uint64_t pointid_t;

struct __align__(8) d_store {
    memalloc_dptr_t memalloc_dptr;
    uint32_t dptr_align_add;
    uint32_t num_points;
    uint64_t start_pointid;
};

class __align__(32) work
    : public work_base {
    public:
    uint64_t num_points;
    uint64_t num_dim;
    uint64_t point_list_nvme_start_byte;

    value_t *clusters;
    value_t *new_clusters;
    uint64_t *membership;
    uint64_t *num_points_in_cluster;
    uint64_t delta;
    value_t diff;
    uint64_t exit_loop_cnt;

    cuda::atomic<int64_t, cuda::thread_scope_device> base_counter;

    // From json
    uint32_t nvme_dev_id;
    uint64_t point_list_nvme_start_GiB;
    uint64_t num_clusters;
    value_t threshold;
    uint64_t num_points_per_warp_per_loop;
    uint64_t max_loop;

    uint32_t prefetch_mult;

    __device__ void work_init_d() {
        delta = 0;
        diff = DBL_MAX;
        base_counter.store(0, cuda::memory_order_relaxed);
        printf("%s: num_points: %lu, num_dim: %lu, num_clusters: %lu\n",
               __func__, num_points, num_dim, num_clusters);
    }
    __device__ void work_fini_d() {
    }

    __host__ void work_init_h(const uint32_t num_controllers, const nlohmann::json &in_js,
                              std::vector<struct bam_array_in> &bam_array_in_vec) {
        static_assert(std::is_standard_layout_v<work> == true);
        static_assert(std::is_trivially_copyable_v<work> == true);

        std::remove_pointer_t<decltype(this)> wd;
        wd.wb_init_h();

        std::string point_filename;
        try {
            point_filename = in_js.at("point_filename");
            wd.nvme_dev_id = in_js.at("nvme_dev_id");
            wd.point_list_nvme_start_GiB = in_js.at("point_list_nvme_start_GiB");
            wd.num_clusters = in_js.at("num_clusters");
            wd.threshold = in_js.at("threshold");
            wd.num_points_per_warp_per_loop = in_js.at("num_points_per_warp_per_loop");
            wd.max_loop = in_js.at("max_loop");

            wd.prefetch_mult = in_js.at("prefetch_mult");
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            force_assert(false);
        }
        wd.point_list_nvme_start_byte = wd.point_list_nvme_start_GiB * 1024UL * 1024UL * 1024UL;

        force_assert(wd.prefetch_mult > 0);

        // Open point metadata
        std::ifstream point_meta(point_filename + ".json");
        if (!point_meta.is_open()) {
            fprintf(stderr, "%s: file open failed.\n", __func__);
            exit(EXIT_FAILURE);
        }
        nlohmann::json js;
        js = nlohmann::json::parse(point_meta);
        point_meta.close();
        wd.num_points = js.at("num_points");
        wd.num_dim = js.at("num_dim");

        // Centroid
        std::vector<value_t> clusters_h(wd.num_clusters * wd.num_dim);
        bool centroid_default = true;
        if (centroid_default) {
            // Use the first num_clusters points from data file
            // as the initial centroids
            std::string data_filename(point_filename);
            data_filename = data_filename + ".data";
            FILE *data_file = fopen(data_filename.c_str(), "r");
            if (!data_file) {
                fprintf(stderr, "Data file open error.\n");
                exit(EXIT_FAILURE);
            }
            size_t ret = fread(clusters_h.data(), sizeof(value_t), wd.num_clusters * wd.num_dim, data_file);
            if (ret != wd.num_clusters * wd.num_dim) {
                fprintf(stderr, "Vector file read error.\n");
                exit(EXIT_FAILURE);
            }
            fclose(data_file);
        } else {
            force_assert(false);
        }
        cudaErrChk(cudaMalloc(&(wd.clusters), wd.num_clusters * wd.num_dim * sizeof(value_t)));
        cudaErrChk(cudaMemcpy(wd.clusters, clusters_h.data(), wd.num_clusters * wd.num_dim * sizeof(value_t), cudaMemcpyHostToDevice));

        cudaErrChk(cudaMalloc(&(wd.new_clusters), wd.num_clusters * wd.num_dim * sizeof(value_t)));
        cudaErrChk(cudaMemset(wd.new_clusters, 0, wd.num_clusters * wd.num_dim * sizeof(value_t)));

        // Other data structures
        cudaErrChk(cudaMalloc(&(wd.membership), wd.num_points * sizeof(uint64_t)));
        cudaErrChk(cudaMemset(wd.membership, 0xFF, wd.num_points * sizeof(uint64_t)));
        cudaErrChk(cudaMalloc(&(wd.num_points_in_cluster), wd.num_clusters * sizeof(uint64_t)));
        cudaErrChk(cudaMemset(wd.num_points_in_cluster, 0, wd.num_clusters * sizeof(uint64_t)));

        memcpy(this, &wd, sizeof(*this));
    }
    __host__ void work_fini_h() {
        std::remove_pointer_t<decltype(this)> wd;
        memcpy(reinterpret_cast<char *>(&wd), this, sizeof(*this));

        cudaErrChk(cudaFree(wd.num_points_in_cluster));
        cudaErrChk(cudaFree(wd.membership));
        cudaErrChk(cudaFree(wd.new_clusters));
        cudaErrChk(cudaFree(wd.clusters));

        wd.wb_fini_h();
    }
};

class __align__(32) work_w
    : public work_base_w<class work> {
    public:
    __host__ void
    work_w_init_h(const uint32_t num_controllers, nlohmann::json in_js);
    __host__ void
    work_w_fini_h();
    __host__ void
    work_print_stats(const uint32_t gpu_clock_mhz, nlohmann::json &out_js);
};

__host__ void
work_w::work_w_init_h(const uint32_t num_controllers, nlohmann::json in_js) {
    wb_w_init_h();
    wd_h.work_init_h(num_controllers, in_js, bam_array_in_vec);
    cudaErrChk(cudaMalloc(&(wd_dp), sizeof(*wd_dp)));
    cudaErrChk(cudaMemcpy(wd_dp, &wd_h, sizeof(*wd_dp), cudaMemcpyHostToDevice));
}

__host__ void
work_w::work_w_fini_h() {
    cudaErrChk(cudaFree(wd_dp));
    wd_h.work_fini_h();
    wb_w_fini_h();
}

__host__ void
work_w::work_print_stats(const uint32_t gpu_clock_mhz, nlohmann::json &out_js) {
    class work wd_temp_h;
    cudaErrChk(cudaMemcpy(&wd_temp_h, wd_dp, sizeof(*wd_dp), cudaMemcpyDeviceToHost));

    uint64_t point_size = wd_temp_h.num_dim * sizeof(value_t);
    printf("%s: point dim: %lu, point size: %lu (bytes), sizeof single point read: %lu\n",
           __func__, wd_temp_h.num_dim, point_size, point_size * wd_temp_h.num_points_per_warp_per_loop);

    float main_cuda_event_js = out_js.at("main_cuda_event_ms");

    uint64_t loop_h = wd_temp_h.exit_loop_cnt;
    value_t diff_h = wd_temp_h.diff;

    float time_per_loop_ms = main_cuda_event_js / loop_h;
    printf("%s: loop count: %lu, last diff: %f, time per loop: %.3lf (ms)\n", __func__, loop_h, diff_h, time_per_loop_ms);

    std::vector<value_t> clusters_h(wd_temp_h.num_clusters * wd_temp_h.num_dim);
    size_t clusters_size = wd_temp_h.num_clusters * wd_temp_h.num_dim * sizeof(value_t);
    cudaErrChk(cudaMemcpy(clusters_h.data(), wd_temp_h.clusters, clusters_size, cudaMemcpyDeviceToHost));

    // Write result to file
    std::string result_filename = "result-kmeans-centroids.data";
    std::remove(result_filename.c_str());
    FILE *fp = fopen(result_filename.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "Result file open error.\n");
        exit(EXIT_FAILURE);
    }
    size_t ret = fwrite(clusters_h.data(), sizeof(value_t), wd_temp_h.num_clusters * wd_temp_h.num_dim, fp);
    if (ret != wd_temp_h.num_clusters * wd_temp_h.num_dim) {
        fprintf(stderr, "Write to file failed.\n");
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    out_js["work_exit_loop_cnt"] = loop_h;
    out_js["work_time_per_loop_ms"] = time_per_loop_ms;
}

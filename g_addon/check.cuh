#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>

class warp_smid {
    public:
    uint32_t *rt_warp_smid;
    uint32_t *app_warp_smid;
    uint32_t rt_len;
    uint32_t app_len;

    __device__ void warp_smid_init_d() {
    }
    __device__ void warp_smid_fini_d() {
    }

    __host__ void warp_smid_init_h(const uint32_t total_num_rt_warps, const uint32_t total_num_app_warps) {
        rt_len = total_num_rt_warps;
        cudaErrChk(cudaMalloc(&(rt_warp_smid), sizeof(uint32_t) * rt_len));
        cudaErrChk(cudaMemset(rt_warp_smid, 0, sizeof(uint32_t) * rt_len));

        app_len = 1024 * 4;
        cudaErrChk(cudaMalloc(&(app_warp_smid), sizeof(uint32_t) * app_len));
        cudaErrChk(cudaMemset(app_warp_smid, 0xFF, sizeof(uint32_t) * app_len));
    }
    __host__ void warp_smid_fini_h() {
        cudaErrChk(cudaFree(app_warp_smid));
        cudaErrChk(cudaFree(rt_warp_smid));
    }

    __device__ void set_rt_warp_smid(const uint32_t wid, const uint32_t smid) {
        rt_warp_smid[wid] = smid;
    }
    __device__ void set_app_warp_smid(const uint32_t wid, const uint32_t smid) {
        force_assert_printf(wid < app_len, "%s: wid: %u, len: %u\n", __func__, wid, app_len);
        app_warp_smid[wid] = smid;
    }

    __host__ void
    check();
};

__host__ void
warp_smid::check() {
    // For each user smid, compare with lib smid
    std::vector<int32_t> rt_smid_h(rt_len);
    std::vector<int32_t> app_smid_h(app_len);
    cudaErrChk(cudaMemcpy(rt_smid_h.data(), rt_warp_smid, sizeof(uint32_t) * rt_smid_h.size(), cudaMemcpyDeviceToHost));
    cudaErrChk(cudaMemcpy(app_smid_h.data(), app_warp_smid, sizeof(uint32_t) * app_smid_h.size(), cudaMemcpyDeviceToHost));

    bool overlap_sm = false;
    for (uint32_t i = 0; i < app_smid_h.size(); ++i) {
        for (uint32_t j = 0; j < rt_smid_h.size(); ++j) {
            // force_assert(app_smid_h[i] != rt_smid_h[j]);
            if (app_smid_h[i] == rt_smid_h[j])
                overlap_sm = true;
        }
    }
    if (overlap_sm) {
        printf("%s: runtime smid: ", __func__);
        for (uint32_t i = 0; i < rt_smid_h.size(); ++i) {
            printf("%u ", rt_smid_h[i]);
        }
        printf("\n");
        printf("%s: app smid: ", __func__);
        for (uint32_t i = 0; i < app_smid_h.size(); ++i) {
            printf("%u ", app_smid_h[i]);
        }
        printf("\n");
    }

    int dev;
    cudaErrChk(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int n_sm = prop.multiProcessorCount;

    std::vector<uint32_t> rt_sm(n_sm);
    std::vector<uint32_t> app_sm(n_sm);
    for (uint32_t i = 0; i < rt_smid_h.size(); ++i)
        rt_sm[rt_smid_h[i]]++;
    for (uint32_t i = 0; i < app_smid_h.size(); ++i) {
        if (app_smid_h[i] >= 0)
            app_sm[app_smid_h[i]]++;
    }

    uint32_t n_rt_sm = 0;
    uint32_t n_app_sm = 0;
    for (size_t i = 0; i < rt_sm.size(); ++i) {
        if (rt_sm[i] != 0)
            n_rt_sm++;
    }
    for (size_t i = 0; i < app_sm.size(); ++i) {
        if (app_sm[i] != 0)
            n_app_sm++;
    }
    printf("%s: # SM used for runtime: %u, app: %u, overlapping? %s\n", __func__,
           n_rt_sm, n_app_sm, overlap_sm ? "true" : "false");
}

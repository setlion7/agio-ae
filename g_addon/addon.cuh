#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda/atomic>
#include <fstream>
#include <iostream>

#include "helper_headers/json.hpp"

#include "buffer.cuh"
#include "memalloc.cuh"
#include "check.cuh"

class __align__(32) addon {
    public:

    bool memalloc_enable;
    class memalloc_w *memalloc_w_h;
    class memalloc *memalloc_d;

    bool lcache_enable;
    class lcache_w *lcache_w_h;
    class lcache *lcache_d;

    bool warp_smid_enable;
    class warp_smid *warp_smid_h;
    class warp_smid *warp_smid_d;

    __device__ void addon_init_d() {
        if (memalloc_enable)
            memalloc_d->memalloc_init_d();
        if (lcache_enable)
            lcache_d->lcache_init_d();
        if (warp_smid_enable)
            warp_smid_d->warp_smid_init_d();
    }
    __device__ void addon_fini_d() {
        if (warp_smid_enable)
            warp_smid_d->warp_smid_fini_d();
        if (memalloc_enable)
            memalloc_d->memalloc_fini_d();
        if (lcache_enable)
            lcache_d->lcache_fini_d();
    }

    __host__ void init_h() {
        static_assert(std::is_standard_layout_v<addon> == true);
        static_assert(std::is_trivially_copyable_v<addon> == true);
    }
    __host__ void fini_h() {
    }
};

class __align__(32) addon_w {
    public:
    class addon *addon_dp;
    class addon *addon_hp;

    uint32_t n_rt_warps;
    uint32_t n_app_warps;

    __host__ void addon_w_init_h(const uint32_t total_num_rt_warps, const uint32_t total_num_app_warps,
                             std::vector<Controller *> ctrls, const nlohmann::json in_js,
                             std::vector<struct bam_array_in> bam_in_vec) {
        static_assert(std::is_standard_layout_v<addon_w> == true);

        addon_hp = (struct addon *)malloc(sizeof(*addon_hp));
        memset(addon_hp, 0, sizeof(*addon_hp));

        n_rt_warps = total_num_rt_warps;
        n_app_warps = total_num_app_warps;

        try {
            addon_hp->memalloc_enable = in_js.at("memalloc").at("enable");
            addon_hp->lcache_enable = in_js.at("cache").at("enable");
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            force_assert(false);
        }
        addon_hp->warp_smid_enable = true;

        if (addon_hp->memalloc_enable) {
            addon_hp->memalloc_w_h = new class memalloc_w;
            addon_hp->memalloc_w_h->memalloc_w_init_h(ctrls[0]->ctrl, in_js.at("memalloc"));
            addon_hp->memalloc_d = addon_hp->memalloc_w_h->memalloc_dp;
        } else {
            addon_hp->memalloc_w_h = nullptr;
            addon_hp->memalloc_d = nullptr;
        }

        if (addon_hp->lcache_enable) {
            addon_hp->lcache_w_h = new class lcache_w;
            addon_hp->lcache_w_h->lcache_w_init_h(ctrls, bam_in_vec, in_js.at("cache"));
            addon_hp->lcache_d = addon_hp->lcache_w_h->lcache_dp;
        } else {
            addon_hp->lcache_w_h = nullptr;
            addon_hp->lcache_d = nullptr;
        }

        if (addon_hp->warp_smid_enable) {
            addon_hp->warp_smid_h = new class warp_smid;
            addon_hp->warp_smid_h->warp_smid_init_h(total_num_rt_warps, total_num_app_warps);
            cudaErrChk(cudaMalloc(&(addon_hp->warp_smid_d), sizeof(class warp_smid)));
            cudaErrChk(cudaMemcpy(addon_hp->warp_smid_d, addon_hp->warp_smid_h, sizeof(class warp_smid), cudaMemcpyHostToDevice));
        }

        addon_hp->init_h();
        cudaErrChk(cudaMalloc(&addon_dp, sizeof(*addon_dp)));
        cudaErrChk(cudaMemcpy(addon_dp, addon_hp, sizeof(*addon_dp), cudaMemcpyHostToDevice));
    }
    __host__ void addon_w_fini_h() {
        addon_hp->fini_h();

        if (addon_hp->warp_smid_enable) {
            addon_hp->warp_smid_h->warp_smid_fini_h();
            delete addon_hp->warp_smid_h;
        }

        if (addon_hp->lcache_enable) {
            addon_hp->lcache_w_h->lcache_w_fini_h();
            delete addon_hp->lcache_w_h;
        }

        // Memalloc
        if (addon_hp->memalloc_enable) {
            addon_hp->memalloc_w_h->memalloc_w_fini_h();
            delete addon_hp->memalloc_w_h;
        }

        free(addon_hp);
        cudaErrChk(cudaFree(addon_dp));
    }

    __host__ void udata_out(const uint32_t gpu_clock_mhz, nlohmann::json &out_js);
};

__host__ void
addon_w::udata_out(const uint32_t gpu_clock_mhz, nlohmann::json &out_js) {
    if (addon_hp->lcache_enable) {
        addon_hp->lcache_w_h->stats_out(out_js);
    }
    if (addon_hp->warp_smid_enable) {
        addon_hp->warp_smid_h->check();
    }
}

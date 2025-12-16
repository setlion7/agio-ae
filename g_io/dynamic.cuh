#pragma once

#include <cuda/atomic>
#include <cuda/semaphore>

#include "ds.cuh"
#include "memalloc.cuh"

class dynamic {
    public:
    cuda::binary_semaphore<cuda::thread_scope_device> sema{1};

    static constexpr long long int update_freq = 1410 * 1000 * 1000;
    // static constexpr long long int update_freq = 1410 * 1000;
    long long int last_check_clk;
    uint64_t n_updates;

    uint32_t n_sq;
    uint32_t n_wq;
    uint32_t sq_threshold;
    uint32_t wq_threshold;

    __device__ void dynamic_init_d(uint32_t n_sq_in, uint32_t n_wq_in, uint32_t sqt, uint32_t wqt) {
        static_assert(std::is_standard_layout_v<dynamic> == true);
        static_assert(std::is_trivially_copyable_v<dynamic> == true);

        n_sq = n_sq_in;
        n_wq = n_wq_in;
        sq_threshold = sqt;
        wq_threshold = wqt;

        last_check_clk = 0;
        n_updates = 0;
    }
    __device__ void dynamic_fini_d() {
    }

    __host__ void dynamic_init_h() {
    }
    __host__ void dynamic_fini_h() {
    }

    __forceinline__ __device__ void
    update() {
        ++n_updates;
        // printf("%s: update every 1 sec (%lu)...\n", __func__, n_updates);
    }
    __forceinline__ __device__ void
    try_update() {
        if (threadIdx.x % 32 == 0) {
            long long int clk = clock64();
            long long int last_clk = last_check_clk;
            long long int diff = clk - last_clk;
            if (diff > update_freq) {
                if (sema.try_acquire()) {
                    if (last_clk == last_check_clk) {
                        update();
                        last_check_clk = clk;
                    }
                    sema.release();
                }
            }
        }
        __syncwarp();
    }
};

#pragma once

#include <cuda.h>
#include <vector>

class __align__(32) work_base {
    public:

    __device__ void wb_init_d() {
    }
    __device__ void wb_fini_d() {
    }
    __host__ void wb_init_h() {
    }
    __host__ void wb_fini_h() {
    }
};

template <typename wd_t>
class work_base_w {
    public:
    wd_t wd_h;
    wd_t *wd_dp;

    std::vector<struct bam_array_in> bam_array_in_vec{};

    __host__ void wb_w_init_h() {
    }
    __host__ void wb_w_fini_h() {
    }
};

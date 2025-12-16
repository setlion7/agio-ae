#pragma once

#include <cuda/std/atomic>
#include <cuda.h>

#include "ds.cuh"
#include "runtime.cuh"
#include "addon.cuh"

#include "work.cuh"

__global__ void
gio_rt_shutdown(class gio *gp) {
    LIB_TID;

    if (tid == 0) {
        gp->shutdown();
        gp->end_value = 0;
    }
    __syncthreads();
}

__global__ void
gio_rt_init(class gio *gp) {
    LIB_TID;

    if (tid == 0) {
        new (gp) class gio;
        gp->gio_init_d();
    }
    __syncthreads();
    __nanosleep(1024);
}

__global__ void
gio_rt_fini(class gio *gp) {
    LIB_TID;

    if (tid == 0) {
        gp->gio_fini_d();
        gp->~gio();
    }
    __syncthreads();
}

__global__ void
addon_kernel_init(class gio *gp, class addon *ap) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        ap->addon_init_d();
    }
    __syncthreads();
}
__global__ void
addon_kernel_fini(class gio *gp, class addon *ap) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        ap->addon_fini_d();
    }
    __syncthreads();
}

__global__ void
app_kernel_init(class gio *gp, class addon *ap, class work *wp) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    force_assert(tid == 0);

    if (tid == 0) {
        gp->mptr = ap->memalloc_d;
        gp->cp = ap->lcache_d;
        gp->wsp = ap->warp_smid_d;
        // force_assert(gp->mptr != nullptr);

        work_init(gp, wp);
    }
}

__global__ void
app_kernel_fini(class gio *gp, class addon *ap, class work *wp) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    force_assert(tid == 0);

    if (tid == 0) {
        work_fini(gp, wp);
    }
}

__global__ void
g_runtime_kernel(class gio *gp, class addon *ap) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t laneId = threadIdx.x & 0x1f;

    uint32_t groupId = blockIdx.x;
    if (laneId == 0) {
        ap->warp_smid_d->set_rt_warp_smid(tid / 32, get_smid2());
    }
    __syncthreads();

    gp->runtime_main(tid, groupId);

    __syncthreads();
}

__global__ void
g_app_kernel_init(class gio *gp) {
    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t laneId = threadIdx.x & 0x1f;

    if (laneId == 0) {
        gp->wsp->set_app_warp_smid(tid / 32, get_smid2());
    }
    __syncthreads();
}

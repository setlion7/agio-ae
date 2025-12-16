#pragma once

#include <cuda.h>

#include "helper_headers/fassert.cuh"

#include "work_ds.cuh"
#include "types_h.cuh"
#include "api.cuh"

__forceinline__ __device__ void
work_init(class gio *gp, class work *wp) {
    wp->work_init_d();

    const uint32_t nvme_num_bytes = wp->num_rows_per_warp_per_loop * wp->num_cols * sizeof(value_t);
    printf("%s: num_rows: %lu, num_cols: %lu, n_rows_per_warp: %lu, nvme read size: %u\n",
           __func__, wp->num_rows, wp->num_cols, wp->num_rows_per_warp_per_loop, nvme_num_bytes);
}

__forceinline__ __device__ void
work_fini(class gio *gp, class work *wp) {
    wp->work_fini_d();
}

__forceinline__ __device__ void
process(const uint64_t start_row, const uint64_t num_rows_per_warp,
        const uint64_t total_num_rows, const uint64_t num_cols, value_t *matrix, value_t *vecX, value_t *vecY) {
    const uint32_t lid = threadIdx.x % 32;
    uint64_t loc = 0;
    for (uint64_t w = start_row; w < start_row + num_rows_per_warp; ++w) {
        // For each row
        __syncwarp();
        if (w >= total_num_rows)
            break;

        value_t sum = 0;
        for (uint64_t i = 0 + lid; i < num_cols; i += 32) {
            sum += matrix[(loc * num_cols) + i] * vecX[i];
        }
        loc += 1;

        // __syncwarp();
        for (int i = 16; i >= 1; i /= 2)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, i, 32);
        // for (int offset = 16; offset > 0; offset /= 2)
        //     sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (lid == 0) {
            vecY[w] = sum;
            // printf("row: %lu done, nnz: %lu, check_duration_us: %lu\n", w, num_nz, (check_end - check_start) / 1410);
        }
    }
    __syncwarp();
}

__forceinline__ __device__ void
process_io(class gio *gp, class work *wp, struct d_store *data) {
    USER_TID;
    uint32_t lid = tid % 32;
    class g_api g(gp);

    value_t *matrix = (value_t *)((char *)data->memalloc_dptr + data->align_add);
    process(data->start_row, wp->num_rows_per_warp_per_loop, wp->num_rows,
            wp->num_cols, matrix, wp->vecX, wp->vecY);
    __syncwarp();

    if (lid == 0) {
        g.free_mem(data->memalloc_dptr);
        g.free_mem(data);
    }
}

__forceinline__ __device__ int32_t
build_iocb(class gio *gp, class work *wp, struct iocb *cb, uint64_t my_row) {
    USER_GID;

    if (my_row >= wp->num_rows) {
        return 1;
    }
    class g_api g(gp);

    uint32_t devid = wp->matrix_nvme_dev_id;
    if (devid > 128) {
        devid = gid % gp->num_ctrls;
    }

    const uint64_t num_rows_per_warp = wp->num_rows_per_warp_per_loop;
    const uint64_t nvme_start_byte = wp->matrix_nvme_start_byte + (my_row * wp->num_cols * sizeof(value_t));
    const uint32_t nvme_num_bytes = num_rows_per_warp * wp->num_cols * sizeof(value_t);

    memalloc_dptr_t mem_dptr = nullptr;
    // if constexpr (true) {
    if constexpr (false) {
        g.get_mem(&mem_dptr, nvme_num_bytes + 512);
        if (mem_dptr == nullptr) {
            return -1;
        }
    }

    struct d_store *data = nullptr;
    g.get_mem((memalloc_dptr_t *)&data, sizeof(*data));
    force_assert_printf(data != nullptr, "size: %lu\n", sizeof(*data));

    data->memalloc_dptr = mem_dptr;
    data->start_row = my_row;

    g.build_rtcb(cb, (uint64_t)data, devid, &(data->memalloc_dptr),
                 nvme_start_byte, nvme_num_bytes, G_API_DIRECT);
    data->align_add = cb->align_add;

    return 0;
}

__global__ void
work_main(class gio *gp, class work *wp, uint32_t issue_read) {
    USER_TID;
    const uint32_t lid = tid % 32;
    const uint32_t wid = tid / 32;
    class g_api g(gp);

    struct iocb cb;

    if (issue_read == 1) {
        uint64_t my_rid;
#if WORK_DYNAMIC == 1
        int64_t base = 0;
        if (lid == 0) {
            base = wp->vid_counter.global_counter(1, cuda::memory_order_relaxed);
        }
        base = __shfl_sync(0xFFFFFFFF, base, 0);
        my_rid = base + lid;

#else // WORK_DYNAMIC == 0
        if (lid == 0)
            my_rid = wid;
        else
            my_rid = 0xFFFFFFFFFFFFFFFFUL;
#endif // WORK_DYNAMIC

        my_rid *= wp->num_rows_per_warp_per_loop;

        if (lid == 0) {
            int32_t ret = build_iocb(gp, wp, &cb, my_rid);
            force_assert(ret >= 0);
        }
    }

    while (true) {
        int32_t status = g.aio_read(&cb);
        // if (__any_sync(0xFFFFFFFF, status == 0)) {
        //     if (lid == 0)
        //         *changed_d = 1;
        // }

        struct d_store *ret_data = nullptr;
        if (lid == 0) {
            ret_data = (struct d_store *)g.wait_any();
        }
        ret_data = (struct d_store *)__shfl_sync(0xFFFFFFFF, (uint64_t)ret_data, 0);
        if ((int64_t)ret_data >= 0) {
            process_io(gp, wp, ret_data);
        }

        bool done = cb.submitted || !cb.valid;
        if (__all_sync(0xFFFFFFFF, done)) {
            break;
        }
    }
}

__global__ void
work_direct_process_only(class gio *gp, class work *wp) {
    USER_TID;
    const uint32_t lid = tid % 32;
    class g_api g(gp);

    struct d_store *ret_data = nullptr;
    if (lid == 0) {
        ret_data = (struct d_store *)g.wait_any();
    }
    ret_data = (struct d_store *)__shfl_sync(0xFFFFFFFF, (uint64_t)ret_data, 0);
    if ((int64_t)ret_data >= 0) {
        process_io(gp, wp, ret_data);
    }
}

__global__ void
work_direct_check(class gio *gp, class work *wp, uint32_t *pending_io) {
    class g_api g(gp);
    int32_t ret = g.check_pending();
    if (ret != 0) {
        *pending_io = 1;
    }
}

#define WORK_KERNEL_NAME work_main

void work_kernel_register() {
    cudaFuncAttributes attr;
    cudaErrChk(cudaFuncGetAttributes(&attr, WORK_KERNEL_NAME));
    cudaErrChk(cudaFuncGetAttributes(&attr, work_direct_process_only));
    cudaErrChk(cudaFuncGetAttributes(&attr, work_direct_check));
}

void work_launcher(nlohmann::json &out_js, const nlohmann::json &cuda_js,
                   class gio_w &gio_wrap, class work_w &work_wrap,
                   class thread_user_data &tu) {
    class gio *gp = gio_wrap.gio_dp;
    class work *wp = work_wrap.wd_dp;
    CUstream *phStream = tu.phStream;

    uint32_t *pending_io_d;
    uint32_t *pending_io_h = new uint32_t;
    cudaErrChk(cudaMallocAsync(&pending_io_d, sizeof(uint32_t), *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));

    // Calculate kernel size
    const uint32_t n_threads = (uint32_t)cuda_js.at("n_app_warps") * 32;
    const uint32_t n_blocks_req = (work_wrap.wd_h.num_rows / (n_threads / 32)) + 1;
    force_assert((uint32_t)cuda_js.at("n_app_blocks") == 0);

    int nB;
    cudaErrChk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nB, WORK_KERNEL_NAME, n_threads, 0));
    int n_SM = cuda_js.at("green_work_n_groups").template get<int>() * cuda_js.at("green_group_size").template get<int>();
    const uint32_t n_blocks_process_only = nB * n_SM;

    printf("%s: launch app kernel n_blocks_req: %u, n_blocks_process_only: %u, n_threads: %u\n",
           __func__, n_blocks_req, n_blocks_process_only, n_threads);
    clock_gettime(CLOCK_REALTIME, &tu.ts_start);
    cudaErrChk(cudaEventRecord(tu.event_start, *phStream));

    uint32_t issue_read = 1;
    uint32_t iter = 0;

    do {
        issue_read = 1;
        void *work_main_args[3] = {&gp, &wp, &issue_read};
        void *work_direct_process_only_args[2] = {&gp, &wp};
        void *work_direct_check_args[3] = {&gp, &wp, &pending_io_d};

    continue_level:
        *pending_io_h = 0;
        cudaErrChk(cudaMemcpyAsync(pending_io_d, pending_io_h, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));
        if (issue_read) {
            cudaErrChk(cudaLaunchKernel((void *)WORK_KERNEL_NAME, n_blocks_req, n_threads, work_main_args, 0, *phStream));
        } else {
            cudaErrChk(cudaLaunchKernel((void *)work_direct_process_only, n_blocks_process_only, n_threads, work_direct_process_only_args, 0, *phStream));
        }

        cudaErrChk(cudaLaunchKernel((void *)work_direct_check, 1, 256, work_direct_check_args, 0, *phStream));
        cudaErrChk(cudaMemcpyAsync(pending_io_h, pending_io_d, sizeof(uint32_t), cudaMemcpyDeviceToHost, *phStream));
        cudaErrChk(cudaStreamSynchronize(*phStream));

        if (*pending_io_h == 1) {
            issue_read = 0;
            goto continue_level;
        }

        break;

        ++iter;
    } while (iter < 10000);

    cudaErrChk(cudaEventRecord(tu.event_end, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    clock_gettime(CLOCK_REALTIME, &tu.ts_end);

    cudaErrChk(cudaFreeAsync(pending_io_d, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    delete pending_io_h;
}

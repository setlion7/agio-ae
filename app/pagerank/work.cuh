#pragma once

#include <cuda/atomic>

#include "work_ds.cuh"
#include "types_h.cuh"
#include "api.cuh"

__forceinline__ __device__ void
work_init(class gio *gp, class work *wp) {
    wp->work_init_d();
}

__forceinline__ __device__ void
work_fini(class gio *gp, class work *wp) {
    wp->work_fini_d();
}

__forceinline__ __device__ int32_t
work::build_iocb(class gio *gp, struct iocb *cb, uint64_t vid) {
    USER_GID;
    if (vid >= vertex_count) {
        return 1;
    }
    if (label[vid] == false) {
        return 1;
    }

    class g_api g(gp);

    const uint64_t start = vertex_list[vid];
    const uint64_t end = vertex_list[vid + 1];
    const uint64_t num_edges = end - start;
    if (num_edges == 0) {
        return 1;
    }

    uint32_t devid = nvme_dev_id;
    if (devid > 128) {
        devid = gid % gp->num_ctrls;
    }

    uint64_t start_byte = nvme_edge_start_byte + (start * sizeof(vertex_t));
    uint64_t n_bytes = num_edges * sizeof(vertex_t);

    memalloc_dptr_t mem_dptr = nullptr;
    // if constexpr (true) {
    if constexpr (false) {
        g.get_mem(&mem_dptr, n_bytes);
        if (mem_dptr == nullptr) {
            return -1;
        }
    }

    struct app_data *data = nullptr;
    g.get_mem((memalloc_dptr_t *)&data, sizeof(*data));
    force_assert_printf(data != nullptr, "%s: sizeof(*data): %lu\n", __func__, sizeof(*data));

    data->parent_start_vid = vid;
    data->num_total_children = num_edges;
    data->child_start_align_add = 0;
    data->memalloc_dptr = mem_dptr;

    g.build_rtcb(cb, (uint64_t)data, devid, &(data->memalloc_dptr), start_byte, n_bytes);
    return 0;
}

__forceinline__ __device__ void
work::process_residual2(class gio *gp, struct app_data *data) {
    USER_TID;
    uint32_t lid = tid % 32;

    vertex_t *edge_ptr = (vertex_t *)((char *)data->memalloc_dptr + data->child_start_align_add);
    uint64_t remainder = (uint64_t)edge_ptr % 128UL;
    uint32_t push_elems = remainder / sizeof(vertex_t);
    vertex_t *shift_ptr = (vertex_t *)((uint64_t)edge_ptr - remainder);

    vertex_t parent_vid = data->parent_start_vid;
    for (uint64_t i = lid; i < data->num_total_children + push_elems; i = i + 32) {
        if (i >= push_elems) {
            vertex_t vid = shift_ptr[i];

            atomicAdd(&residual[vid], delta[parent_vid]);
            label[parent_vid] = false;
        }
    }
    __syncwarp();
}

__forceinline__ __device__ void
work::update_func_ext(class gio *gp, vertex_t vid, uint32_t *changed) {
    if (vid < vertex_count) {
        if (residual[vid] > tolerance) {
            value[vid] += residual[vid];

            uint64_t num_edges = vertex_list[vid + 1] - vertex_list[vid];
            if (num_edges) {
                delta[vid] = residual[vid] * alpha / num_edges;
            } else
                delta[vid] = 0.0f;
            residual[vid] = 0.0f;
            label[vid] = true;
            *changed = true;
        }
    }
}

__global__ void
work_main(class gio *gp, class work *wp, uint32_t issue_read) {
    USER_TID;
    const uint32_t lid = tid % 32;
    class g_api g(gp);

    struct iocb cb;

    if (issue_read == 1) {
        vertex_t my_vid;
#if WORK_DYNAMIC == 1
        int64_t base = 0;
        if (lid == 0) {
            base = wp->vid_counter.fetch_add(32, cuda::memory_order_relaxed);
        }
        base = __shfl_sync(0xFFFFFFFF, base, 0);
        my_vid = base + lid;

#else // WORK_DYNAMIC == 0
        my_vid = tid;
#endif // WORK_DYNAMIC

        int32_t ret = wp->build_iocb(gp, &cb, my_vid);
        if (ret < 0) {
            force_assert(false);
        }
    }

    while (true) {
        int32_t status = g.aio_read(&cb);

        struct app_data *ret_data = nullptr;
        if (lid == 0) {
            ret_data = (struct app_data *)g.wait_any();
        }
        ret_data = (struct app_data *)__shfl_sync(0xFFFFFFFF, (uint64_t)ret_data, 0);
        if ((int64_t)ret_data >= 0) {
            wp->process_residual2(gp, ret_data);

            if (lid == 0) {
                g.free_mem(ret_data->memalloc_dptr);
                g.free_mem(ret_data);
            }
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

    struct app_data *ret_data = nullptr;
    if (lid == 0) {
        ret_data = (struct app_data *)g.wait_any();
    }
    ret_data = (struct app_data *)__shfl_sync(0xFFFFFFFF, (uint64_t)ret_data, 0);
    if ((int64_t)ret_data >= 0) {
        wp->process_residual2(gp, ret_data);

        if (lid == 0) {
            g.free_mem(ret_data->memalloc_dptr);
            g.free_mem(ret_data);
        }
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

__global__ void
update(class gio *gp, class work *wp, uint32_t *changed_d) {
    USER_TID;
    wp->update_func_ext(gp, tid, changed_d);
}

#define WORK_KERNEL_NAME work_main

void work_kernel_register() {
    cudaFuncAttributes attr;
    cudaErrChk(cudaFuncGetAttributes(&attr, WORK_KERNEL_NAME));
    cudaErrChk(cudaFuncGetAttributes(&attr, work_direct_process_only));
    cudaErrChk(cudaFuncGetAttributes(&attr, work_direct_check));
    cudaErrChk(cudaFuncGetAttributes(&attr, update));
}

void work_launcher(nlohmann::json &out_js, const nlohmann::json &cuda_js,
                   class gio_w &gio_wrap, class work_w &work_wrap,
                   class thread_user_data &tu) {
    class gio *gp = gio_wrap.gio_dp;
    class work *wp = work_wrap.wd_dp;
    CUstream *phStream = tu.phStream;

    uint32_t *changed_d;
    uint32_t *changed_h = new uint32_t;
    cudaErrChk(cudaMallocAsync(&changed_d, sizeof(uint32_t), *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));

    uint32_t *pending_io_d;
    uint32_t *pending_io_h = new uint32_t;
    cudaErrChk(cudaMallocAsync(&pending_io_d, sizeof(uint32_t), *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));

    const uint32_t max_level = work_wrap.wd_h.max_level;
    uint32_t local_level = 0;
    uint32_t issue_read = 1;

    // Calculate kernel size
    const uint32_t n_threads = (uint32_t)cuda_js.at("n_app_warps") * 32;
    const uint32_t n_blocks_req = (work_wrap.wd_h.vertex_count / n_threads) + 1;
    force_assert((uint32_t)cuda_js.at("n_app_blocks") == 0);

    int nB;
    cudaErrChk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nB, work_direct_process_only, n_threads, 0));
    int n_SM = cuda_js.at("green_work_n_groups").template get<int>() * cuda_js.at("green_group_size").template get<int>();
    const uint32_t n_blocks_process_only = nB * n_SM;

    G_BENCH_CPU_APP_START();

    printf("%s: launch app kernel n_blocks_req: %u, n_blocks_process_only: %u, n_threads: %u\n",
           __func__, n_blocks_req, n_blocks_process_only, n_threads);
    clock_gettime(CLOCK_REALTIME, &tu.ts_start);
    cudaErrChk(cudaEventRecord(tu.event_start, *phStream));

    do {
        issue_read = 1;
        void *work_main_args[3] = {&(gp), &(wp), &issue_read};
        void *work_direct_process_only_args[2] = {&(gp), &(wp)};
        void *work_direct_check_args[3] = {&(gp), &(wp), &pending_io_d};
        G_BENCH_CPU_LEVEL_START();

        *changed_h = 0;
        cudaErrChk(cudaMemcpyAsync(changed_d, changed_h, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));

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

        update<<<n_blocks_req, n_threads, 0, *phStream>>>(gp, wp, changed_d);
        cudaErrChk(cudaMemcpyAsync(changed_h, changed_d, sizeof(uint32_t), cudaMemcpyDeviceToHost, *phStream));
        cudaErrChk(cudaStreamSynchronize(*phStream));

        G_BENCH_CPU_LEVEL_END(out_js, local_level);

        if (*changed_h == 0) {
            break;
        }

        ++local_level;

    } while (local_level < max_level);

    cudaErrChk(cudaEventRecord(tu.event_end, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    clock_gettime(CLOCK_REALTIME, &tu.ts_end);

    out_js.emplace("n_levels", local_level);
    if (local_level == max_level) {
        printf(CRED "%s: exit loop due to reaching max level: %u\n" CRESET, __func__, local_level);
    }

    cudaErrChk(cudaFreeAsync(pending_io_d, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    delete pending_io_h;

    cudaErrChk(cudaFreeAsync(changed_d, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    delete changed_h;
}

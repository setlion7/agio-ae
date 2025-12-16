#pragma once

#include <cuda/atomic>
#include <cuda.h>

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

// 0: build successful
// 1: no-op
// -1: retry later
__forceinline__ __device__ int32_t
build_iocb(class gio *gp, class work *wp,
           struct iocb *cb1, struct iocb *cb2, const vertex_t vid, const int32_t level) {
    USER_GID;

    if (vid >= wp->vertex_count) {
        return 1;
    }

    if (wp->level[vid] != level) {
        return 1;
    }

    class g_api g(gp);

    const vertex_t start = wp->vertex_list[vid];
    const vertex_t end = wp->vertex_list[vid + 1];

    const uint64_t num_edges = end - start;
    if (num_edges == 0) {
        return 1;
    }
    uint32_t devid = wp->nvme_dev_id;
    if (devid > 128) {
        devid = gid % gp->num_ctrls;
    }

    uint64_t edge_nvme_start_byte = wp->edge_list_nvme_start_byte + (start * sizeof(vertex_t));
    uint64_t value_nvme_start_byte = wp->value_list_nvme_start_byte + (start * sizeof(value_t));
    uint32_t edge_bytes_to_read = num_edges * sizeof(vertex_t);
    uint32_t value_bytes_to_read = num_edges * sizeof(value_t);

    // Memory
    memalloc_dptr_t edge_dptr_memalloc = nullptr;
    memalloc_dptr_t value_dptr_memalloc = nullptr;
    // if constexpr (true) {
    if constexpr (false) {
        g.get_mem(&edge_dptr_memalloc, edge_bytes_to_read);
        if (edge_dptr_memalloc == nullptr) {
            return -1;
        }
        g.get_mem(&value_dptr_memalloc, value_bytes_to_read);
        if (value_dptr_memalloc == nullptr) {
            g.free_mem(edge_dptr_memalloc);
            return -1;
        }
    }

    struct app_data *data;
    g.get_mem((void **)&data, sizeof(*data));
    force_assert(data != nullptr);

    data->parent_start_vid = vid;
    data->num_total_children = num_edges;
    data->value_cid = (uint64_t)data + 1;

    data->edge_dptr_memalloc = edge_dptr_memalloc;
    data->value_dptr_memalloc = value_dptr_memalloc;
    data->edge_start_align_add = 0;
    data->value_start_align_add = 0;

    if (level >= 0) {
        data->level = level;
        wp->update = 1;
    }

    g.build_rtcb(cb1, (uint64_t)data, devid, &(data->edge_dptr_memalloc), edge_nvme_start_byte, edge_bytes_to_read);
    g.build_rtcb(cb2, (uint64_t)data + 1, devid, &(data->value_dptr_memalloc), value_nvme_start_byte, value_bytes_to_read, G_API_NO_NOTIFY);
    return 0;
}

// Given a struct app_data, process the data
__forceinline__ __device__ void
process_cost_direct(class gio *gp, class work *wptr, struct app_data *data) {
    USER_TID;
    const uint32_t lid = tid % 32;

    vertex_t *edge_ptr = (vertex_t *)((char *)data->edge_dptr_memalloc + data->edge_start_align_add);
    value_t *value_ptr = (value_t *)((char *)data->value_dptr_memalloc + data->value_start_align_add);
    uint64_t num_children = data->num_total_children;

    vertex_t parent_vid = data->parent_start_vid;
    const value_t cost = wptr->cost_list[parent_vid].load(cuda::memory_order_relaxed);

    for (uint64_t i = 0 + lid; i < num_children; i = i + 32) {
        vertex_t vid = edge_ptr[i];
        value_t weight = value_ptr[i];

        force_assert_printf(vid < wptr->vertex_count, "%s: vid: %lu, dstore: %p\n", __func__, vid, data);
        wptr->cost_list[vid].fetch_min(cost + weight, cuda::memory_order_relaxed);
    }
}

__global__ void
work_main(class gio *gp, class work *wptr,
          const uint32_t local_level, const uint32_t issue_read) {
    USER_TID;
    const uint32_t lid = tid % 32;

    class g_api g(gp);
    struct iocb cb[2];

    if (issue_read == 1) {
        vertex_t my_vid;
#if WORK_DYNAMIC == 1
        int64_t base = 0;
        if (lid == 0) {
            base = wptr->vid_counter.fetch_add(32, cuda::memory_order_relaxed);
        }
        base = __shfl_sync(0xFFFFFFFF, base, 0);
        my_vid = base + lid;
#else  // WORK_DYNAMIC == 0
        my_vid = tid;
#endif // WORK_DYNAMIC

        int32_t ret = build_iocb(gp, wptr, &cb[0], &cb[1], my_vid, local_level);
        if (ret < 0) {
            force_assert(false);
        }
    }

    while (true) {
        g.aio_read_n(cb, 2);

        struct app_data *ret_data = nullptr;
        if (lid == 0) {
            ret_data = (struct app_data *)g.wait_any();
        }
        ret_data = (struct app_data *)__shfl_sync(0xFFFFFFFF, (uint64_t)ret_data, 0);
        if ((int64_t)ret_data > 0) {
            process_cost_direct(gp, wptr, ret_data);
            __syncwarp();

            if (lid == 0) {
                g.free_mem(ret_data->edge_dptr_memalloc);
                g.free_mem(ret_data->value_dptr_memalloc);
                g.free_mem(ret_data);
            }
        }

        bool done = (cb[0].submitted || !cb[0].valid) && (cb[1].submitted || !cb[1].valid);
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
    if ((int64_t)ret_data > 0) {
        process_cost_direct(gp, wp, ret_data);
        __syncwarp();

        if (lid == 0) {
            g.free_mem(ret_data->edge_dptr_memalloc);
            g.free_mem(ret_data->value_dptr_memalloc);
            g.free_mem(ret_data);
        }
    }
}

__global__ void
update(class gio *gp, class work *wptr, const uint32_t local_level, uint32_t *changed) {
    USER_TID;
    if (tid == 0) {
        wptr->vid_counter.store(0, cuda::memory_order_relaxed);
    }

    if (tid < wptr->vertex_count) {
        uint32_t cost = wptr->cost_list[tid].load(cuda::memory_order_relaxed);
        if (cost < wptr->new_cost_list[tid].load(cuda::memory_order_relaxed)) {
            wptr->new_cost_list[tid].store(cost, cuda::memory_order_relaxed);
            wptr->level[tid] = local_level + 1;
            *changed = 1;
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
    const uint32_t n_blocks_issue = (work_wrap.wd_h.vertex_count / n_threads) + 1;
    // const uint32_t n_blocks = (wdata_wrap.wd_h_init.vertex_count / (n_threads / 32)) + 1; // warp-only issue
    force_assert((uint32_t)cuda_js.at("n_app_blocks") == 0);

    int nB;
    cudaErrChk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nB, work_direct_process_only, n_threads, 0));
    int n_SM = cuda_js.at("green_work_n_groups").template get<int>() * cuda_js.at("green_group_size").template get<int>();
    const uint32_t n_blocks_process_only = nB * n_SM;

    // Init
    const uint32_t zero = 0;
    cudaErrChk(cudaMemcpyAsync(&work_wrap.wd_h.level[work_wrap.wd_h.root], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));
    cudaErrChk(cudaMemcpyAsync(&work_wrap.wd_h.cost_list[work_wrap.wd_h.root], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));
    cudaErrChk(cudaMemcpyAsync(&work_wrap.wd_h.new_cost_list[work_wrap.wd_h.root], &zero, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));

    G_BENCH_CPU_APP_START();

    printf("%s: launch app kernel n_blocks_issue: %u, n_blocks_process_only: %u, n_threads: %u\n",
           __func__, n_blocks_issue, n_blocks_process_only, n_threads);
    clock_gettime(CLOCK_REALTIME, &tu.ts_start);
    cudaErrChk(cudaEventRecord(tu.event_start, *phStream));

    do {
        issue_read = 1;
        void *work_main_args[4] = {&(gp), &(wp), &local_level, &issue_read};
        void *work_direct_process_only_args[2] = {&(gp), &(wp)};
        void *work_direct_check_args[3] = {&(gp), &(wp), &pending_io_d};
        G_BENCH_CPU_LEVEL_START();

    continue_level:
        // printf("%s: launch direct kernel %u * %u..., local_level: %u, issue_read: %u\n",
        //        __func__, n_blocks, n_threads, local_level, issue_read);
        *pending_io_h = 0;
        cudaErrChk(cudaMemcpyAsync(pending_io_d, pending_io_h, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));
        if (issue_read) {
            cudaErrChk(cudaLaunchKernel((void *)WORK_KERNEL_NAME, n_blocks_issue, n_threads, work_main_args, 0, *phStream));
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

        *changed_h = 0;
        cudaErrChk(cudaMemcpyAsync(changed_d, changed_h, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));
        update<<<n_blocks_issue, n_threads, 0, *phStream>>>(gp, wp, local_level, changed_d);
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

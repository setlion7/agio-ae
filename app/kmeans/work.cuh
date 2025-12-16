#pragma once

#include <cuda.h>

#include "helper_headers/fassert.cuh"

#include "work_ds.cuh"
#include "types_h.cuh"
#include "api.cuh"

__forceinline__ __device__ void
print_centroids(class work *wp, uint64_t n_initial_centroids) {
    // Print to screen
    printf("%s: centroids (printing initial %lu): \n", __func__, n_initial_centroids);

    uint64_t n_cent = n_initial_centroids;
    if (n_initial_centroids > wp->num_clusters)
        n_cent = wp->num_clusters;

    for (uint64_t i = 0; i < n_cent; ++i) {
        printf("%lu: ", i);
        for (uint64_t j = 0; j < wp->num_dim; ++j)
            printf("%.3f ", wp->clusters[i * wp->num_dim + j]);
        printf("\n");
    }
}

__forceinline__ __device__ void
work_init(class gio *gp, class work *wp) {
    wp->work_init_d();
}

__forceinline__ __device__ void
work_fini(class gio *gp, class work *wp) {
    print_centroids(wp, 5);

    wp->work_fini_d();
}

__forceinline__ __device__ void
update(uint32_t f_tid, class work *wp) {
    // Set data structures to zero
    if (f_tid == 0) {
#if WORK_HOST_MEM == 0
        my_memcpy(wp->clusters, wp->new_clusters, wp->num_clusters * wp->num_dim * sizeof(value_t));

        memset(wp->new_clusters, 0, wp->num_clusters * wp->num_dim * sizeof(value_t));
        memset(wp->num_points_in_cluster, 0, wp->num_clusters * sizeof(uint64_t));
#endif

        wp->diff = (value_t)wp->delta / wp->num_points;
        // printf("%s: f_tid: %u, threshold: %f, delta: %lu, diff: %f, clusters: %f %f %f %f\n",
        //        __func__, f_tid, wptr->threshold, wptr->delta, wptr->diff,
        //        wptr->clusters[0], wptr->clusters[1], wptr->clusters[2], wptr->clusters[3]);

        wp->delta = 0;
        cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);

        wp->base_counter.store(0, cuda::memory_order_relaxed);
    }
}

// Calculate the euclidean distance between point and cluster
__forceinline__ __device__ value_t
edist(value_t *points, value_t *clusters, const uint64_t num_dim,
      const uint64_t pointId, const uint64_t clusterId) {
    value_t result = 0.0;
    for (uint64_t i = 0; i < num_dim; ++i) {
        result += (points[num_dim * pointId + i] - clusters[num_dim * clusterId + i]) *
                  (points[num_dim * pointId + i] - clusters[num_dim * clusterId + i]);
    }
    return result;
}

__forceinline__ __device__ value_t
edist_single(const value_t *point, const value_t *cluster, const uint64_t num_dim) {
    value_t result = 0.0;
    for (uint64_t i = 0; i < num_dim; ++i) {
        result += (point[i] - cluster[i]) * (point[i] - cluster[i]);
    }
    return result;
}

// Aggregate atomic increments to the same location per warp
template <typename T>
__forceinline__ __device__ T
atomicAggInc(T *ptr) {
    int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
    int leader = __ffs(mask) - 1;
    T res;
    if (lane_id() == leader)
        res = atomicAdd(ptr, __popc(mask));
    res = __shfl_sync(mask, res, leader);
    return res + __popc(mask & ((1 << lane_id()) - 1));
}

__forceinline__ __device__ void
process1(class work *wp, value_t *points, const uint64_t start_point_id, const uint64_t num_points_per_warp) {
    const uint32_t lid = threadIdx.x % 32;

    const uint64_t num_dim = wp->num_dim;

    uint64_t loc = 0 + lid;
    for (uint64_t pid = start_point_id + lid; pid < start_point_id + num_points_per_warp;) {
        if (pid >= wp->num_points)
            break;

        // For each point, get distance to each centroid
        // and find the closest cluster
        uint64_t min_dist_clusterId = UINT64_MAX;
        value_t min_dist = VALUE_T_MAX;
        value_t dist;
        for (uint64_t i = 0; i < wp->num_clusters; ++i) {
            dist = edist_single(&(points[loc * num_dim]), &(wp->clusters[i * num_dim]), num_dim);
            if (dist < min_dist) {
                min_dist = dist;
                min_dist_clusterId = i;
            }
        }
        uint32_t membership_changed = 0;
        if (wp->membership[pid] != min_dist_clusterId)
            membership_changed = 1;
        wp->membership[pid] = min_dist_clusterId;

        if (membership_changed == 1)
            // atomicAggInc<unsigned long long int>((unsigned long long int *)&(wptr->delta));
            atomicAdd((unsigned long long int *)&(wp->delta), 1);

        // Reduce (first part) - calculate new cluster centroid coordinates
        atomicAdd((unsigned long long int *)&(wp->num_points_in_cluster[min_dist_clusterId]), 1);
        for (uint64_t i = 0; i < num_dim; ++i) {
            value_t r = points[loc * num_dim + i];
            atomicAdd(&(wp->new_clusters[min_dist_clusterId * num_dim + i]), r);
        }

        pid += 32;
        loc += 32;
    }
}

__forceinline__ __device__ void
process2(class work *wp, const uint64_t my_clusterId) {
    // For each cluster, divide by the number of members of that cluster
    // (to get the new cluster coordinates)
    for (uint64_t j = 0; j < wp->num_dim; ++j)
        wp->new_clusters[my_clusterId * wp->num_dim + j] /= (value_t)wp->num_points_in_cluster[my_clusterId];
}

// Return 0: build successful, 1: no-op, -1: retry later
__forceinline__ __device__ int32_t
build_iocb(class gio *gp, class work *wp,
           class iocb *cb, const uint64_t start_pointid, const uint32_t num_points_in) {
    USER_GID;
    class g_api g(gp);

    if (start_pointid >= wp->num_points)
        return 1;
    const uint64_t point_size = wp->num_dim * sizeof(value_t);

    uint32_t devid = wp->nvme_dev_id;
    if (devid > 128) {
        devid = gid % gp->num_ctrls;
    }

    uint64_t start_byte = wp->point_list_nvme_start_byte + (start_pointid * point_size);
    uint32_t read_size = num_points_in * point_size;

    memalloc_dptr_t mem_dptr = nullptr;
    if constexpr (false) {
        g.get_mem(&mem_dptr, read_size + 1024);
        if (mem_dptr == nullptr) {
            return -1;
        }
    }

    struct d_store *data;
    g.get_mem((memalloc_dptr_t *)&data, sizeof(*data));
    force_assert(data != nullptr);

    data->memalloc_dptr = mem_dptr;
    data->start_pointid = start_pointid;
    data->num_points = num_points_in;

    g.build_rtcb(cb, (uint64_t)data, devid, &(data->memalloc_dptr), start_byte, read_size, G_API_DIRECT);
    data->dptr_align_add = cb->align_add;

    return 0;
}

__global__ void
work_main(class gio *gp, class work *wp, uint32_t issue_read) {
    USER_TID;
    const uint32_t wid = tid / 32;
    const uint32_t lid = tid % 32;

    class g_api g(gp);

    const uint32_t mult = wp->prefetch_mult;
    force_assert(mult <= WORK_MAX_MULT);

    struct iocb cb[WORK_MAX_MULT];

    uint32_t n_points = wp->num_points_per_warp_per_loop;

    if ((issue_read == 1) && (lid == 0)) {
        pointid_t my_point;
#if WORK_DYNAMIC_POINT == 1
        my_point = wp->base_counter.fetch_add(n_points * mult, cuda::memory_order_relaxed);
#else
        my_point = n_points * wid * mult;
#endif
        for (uint32_t i = 0; i < mult; ++i) {
            int32_t ret = build_iocb(gp, wp, &cb[i], my_point + (n_points * i), n_points);
            force_assert(ret >= 0);
        }
    }

    while (true) {
        for (uint32_t i = 0; i < mult; ++i) {
            int32_t status = g.aio_read(&cb[i]);
        }

        struct d_store *ret_data = nullptr;
        value_t *points;
        if (lid == 0) {
            ret_data = (struct d_store *)g.wait_any();
        }
        ret_data = (struct d_store *)__shfl_sync(0xFFFFFFFF, (uint64_t)ret_data, 0);
        if ((int64_t)ret_data >= 0) {
            points = (value_t *)((char *)ret_data->memalloc_dptr + ret_data->dptr_align_add);
            process1(wp, points, ret_data->start_pointid, n_points);
            __syncwarp();

            if (lid == 0) {
                g.free_mem(ret_data->memalloc_dptr);
                g.free_mem(ret_data);
            }
        }

        bool done = true;
        for (uint32_t i = 0; i < mult; ++i) {
            done = cb[i].submitted || !cb[i].valid;
            if (done == false)
                break;
        }
        if (__all_sync(0xFFFFFFFF, done)) {
            break;
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
process2_wrapper(class gio *gp, class work *wp) {
    USER_TID;
    if (tid < wp->num_clusters) {
        process2(wp, tid);
    }
}

__global__ void
update_wrapper(class gio *gp, class work *wp) {
    update(0, wp);
}

#define WORK_KERNEL_NAME work_main

void work_kernel_register() {
    cudaFuncAttributes attr;
    cudaErrChk(cudaFuncGetAttributes(&attr, WORK_KERNEL_NAME));
    cudaErrChk(cudaFuncGetAttributes(&attr, work_direct_check));
    cudaErrChk(cudaFuncGetAttributes(&attr, process2_wrapper));
    cudaErrChk(cudaFuncGetAttributes(&attr, update_wrapper));
}

void work_launcher(nlohmann::json &out_js, const nlohmann::json &cuda_js,
                   class gio_w &gpuio_wrap, class work_w &wdata_wrap,
                   class thread_user_data &tu) {
    class gio *gp = gpuio_wrap.gio_dp;
    class work *wp = wdata_wrap.wd_dp;
    CUstream *phStream = tu.phStream;

    uint32_t *pending_io_d;
    uint32_t *pending_io_h = new uint32_t;
    cudaErrChk(cudaMallocAsync(&pending_io_d, sizeof(uint32_t), *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));

    const uint32_t max_level = wdata_wrap.wd_h.max_loop;

    uint32_t level = 0;
    uint32_t issue_read = 1;

    // Calculate kernel size
    const uint64_t n_points_per_warp = wdata_wrap.wd_h.num_points_per_warp_per_loop;
    const uint64_t n_cu = (wdata_wrap.wd_h.num_points + n_points_per_warp - 1) / n_points_per_warp;

    const uint32_t n_threads = (uint32_t)cuda_js.at("n_app_warps") * 32;
    // const uint32_t n_blocks = (wdata_wrap.wd_h.num_points / n_threads) + 1;
    const uint32_t n_blocks = (n_cu + (n_threads / 32) - 1) / (n_threads / 32);
    force_assert((uint32_t)cuda_js.at("n_app_blocks") == 0);

    value_t diff_h = std::numeric_limits<value_t>::max();
    size_t num_clusters = wdata_wrap.wd_h.num_clusters;

    clock_gettime(CLOCK_REALTIME, &tu.ts_start);
    cudaErrChk(cudaEventRecord(tu.event_start, *phStream));

    do {
        issue_read = 1;
        void *work_main_args[3] = {&(gp), &(wp), &issue_read};
        void *work_direct_check_args[3] = {&(gp), &(wp), &pending_io_d};
        printf("%s: launch direct kernel %u * %u..., level: %u\n", __func__, n_blocks, n_threads, level);

    continue_level:
        *pending_io_h = 0;
        cudaErrChk(cudaMemcpyAsync(pending_io_d, pending_io_h, sizeof(uint32_t), cudaMemcpyHostToDevice, *phStream));

        cudaErrChk(cudaLaunchKernel((void *)WORK_KERNEL_NAME, n_blocks, n_threads, work_main_args, 0, *phStream));

        cudaErrChk(cudaLaunchKernel((void *)work_direct_check, 1, 256, work_direct_check_args, 0, *phStream));
        cudaErrChk(cudaMemcpyAsync(pending_io_h, pending_io_d, sizeof(uint32_t), cudaMemcpyDeviceToHost, *phStream));
        cudaErrChk(cudaStreamSynchronize(*phStream));

        if (*pending_io_h == 1) {
            issue_read = 0;
            goto continue_level;
        }

        process2_wrapper<<<(num_clusters / 256) + 1, 256, 0, *phStream>>>(gp, wp);

        update_wrapper<<<1, 1, 0, *phStream>>>(gp, wp);

#if WORK_HOST_MEM == 1
        // Memcpy from host?
        size_t clusters_size = wdata_wrap.wd_h.num_clusters * wdata_wrap.wd_h.num_dim * sizeof(value_t);
        cudaErrChk(cudaMemcpyAsync(wdata_wrap.wd_h.clusters, wdata_wrap.wd_h.new_clusters, clusters_size, cudaMemcpyDeviceToDevice, *phStream));
        cudaErrChk(cudaMemsetAsync(wdata_wrap.wd_h.new_clusters, 0, clusters_size, *phStream));
        cudaErrChk(cudaMemsetAsync(wdata_wrap.wd_h.num_points_in_cluster, 0, wdata_wrap.wd_h.num_clusters * sizeof(uint64_t), *phStream));
#endif

        cudaErrChk(cudaMemcpyAsync(&diff_h, &(wp->diff), sizeof(diff_h), cudaMemcpyDeviceToHost, *phStream));
        cudaErrChk(cudaStreamSynchronize(*phStream));

        printf("%s: level: %u, diff_h: %f\n", __func__, level, diff_h);
        ++level;

    } while (level < max_level && diff_h > wdata_wrap.wd_h.threshold);

    cudaErrChk(cudaEventRecord(tu.event_end, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    clock_gettime(CLOCK_REALTIME, &tu.ts_end);

    uint64_t loop_cnt = level;
    cudaErrChk(cudaMemcpyAsync(&wp->exit_loop_cnt, &loop_cnt, sizeof(loop_cnt), cudaMemcpyHostToDevice, *phStream));

    cudaErrChk(cudaFreeAsync(pending_io_d, *phStream));
    cudaErrChk(cudaStreamSynchronize(*phStream));
    delete pending_io_h;
}

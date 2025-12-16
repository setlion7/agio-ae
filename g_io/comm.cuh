#pragma once

#include <cuda/atomic>

#include "lqueue.cuh"
#include "macro.h"
#include "types.cuh"

typedef struct padded_atomic<uint32_t> prp_next_t;
typedef struct padded_atomic_status<uint32_t> prp_dma_status_t;
typedef uint32_t sq_ptr_t;
typedef uint32_t sq_status_t;
typedef uint32_t wq_ptr_t;
typedef uint32_t wq_status_t;

struct __align__(32) prp_pool {
    prp_next_t prp_next_dma_page;
    prp_dma_status_t *prp_dma_status;
    uint64_t prp_dma_buffer_ioaddr;
    void *prp_dma_buffer_dptr;
    uint64_t prp_dma_num_pages;

    __device__ void prp_pool_init_d() {
        prp_next_dma_page.value.store(0);
    }
    __device__ void prp_pool_fini_d() {
    }

    __host__ void prp_pool_init_h(uint64_t n_pages, uint64_t dma_ioaddr, void *dma_dptr) {
        static_assert(std::is_standard_layout_v<prp_pool> == true);
        static_assert(std::is_trivially_copyable_v<prp_pool> == true);

        std::vector<prp_dma_status_t> temp3_h(n_pages);
        for (size_t j = 0; j < temp3_h.size(); ++j) {
            temp3_h[j].value = ENTRY_EMPTY;
        }

        std::remove_pointer_t<decltype(this)> pp;
        cudaErrChk(cudaMalloc(&(pp.prp_dma_status), sizeof(prp_dma_status_t) * n_pages));
        cudaErrChk(cudaMemcpy(pp.prp_dma_status, temp3_h.data(), sizeof(prp_dma_status_t) * n_pages, cudaMemcpyHostToDevice));
        pp.prp_dma_num_pages = n_pages;
        pp.prp_dma_buffer_ioaddr = dma_ioaddr;
        pp.prp_dma_buffer_dptr = dma_dptr;
        cudaErrChk(cudaMemset(pp.prp_dma_buffer_dptr, 0, n_pages * 4096UL));
        memcpy(this, &pp, sizeof(*this));
    }
    __host__ void prp_pool_fini_h() {
        std::remove_pointer_t<decltype(this)> pp;
        memcpy(reinterpret_cast<char *>(&pp), this, sizeof(*this));
        cudaErrChk(cudaFree(pp.prp_dma_status));
    }
};

struct __align__(64) io_s_entry {
    uint64_t command_id;
    uint64_t address;
    uint64_t start_byte;
    uint64_t num_bytes;
    gpua_opcode_t opcode;
};

struct __align__(32) s_queue {
    class lqueue<struct io_s_entry, sq_status_t, sq_ptr_t, n_sq_entries> lq;
    cuda::atomic<uint32_t, cuda::thread_scope_device> current_n_items;

    cuda::atomic<uint32_t, cuda::thread_scope_device> n_thresh_exceeded;
    cuda::atomic<uint32_t, cuda::thread_scope_device> n_cmp_false;
    cuda::atomic<uint32_t, cuda::thread_scope_device> qstate;
    uint32_t my_idx;

    // Additional PRP per queue
    struct prp_pool prp_dma_pool;

    __device__ void s_queue_init_d(uint32_t idx, uint32_t sq_thresh) {
        lq.lqueue_init_d(sq_thresh);
        prp_dma_pool.prp_pool_init_d();
        my_idx = idx;
        current_n_items.store(0);
        n_thresh_exceeded.store(0);
        n_cmp_false.store(0);
        qstate.store(1);
    }
    __device__ void s_queue_fini_d() {
        printf("%s: sq[%u]: %u %u, prp_next: %u, n_cmp_false: %u, n_thresh_exceeded: %u\t",
               __func__, my_idx,
               lq.head.value.load(),
               lq.tail.value.load(),
               prp_dma_pool.prp_next_dma_page.value.load(),
               n_cmp_false.load(),
               n_thresh_exceeded.load());
        if (my_idx % 2 == 1)
            printf("\n");
        prp_dma_pool.prp_pool_fini_d();
        lq.lqueue_fini_d();
    }

    __host__ void s_queue_init(uint32_t n_queue_entries,
                               uint64_t n_pages, uint64_t dma_ioaddr, void *dma_dptr) {
        static_assert(std::is_standard_layout_v<s_queue> == true);
        static_assert(std::is_trivially_copyable_v<s_queue> == true);

        decltype(lq) lq_temp_h;
        lq_temp_h.lqueue_init_h();
        cudaErrChk(cudaMemcpy(&lq, &lq_temp_h, sizeof(lq), cudaMemcpyHostToDevice));

        decltype(prp_dma_pool) prp_temp_h;
        prp_temp_h.prp_pool_init_h(n_pages, dma_ioaddr, dma_dptr);
        cudaErrChk(cudaMemcpy(&prp_dma_pool, &prp_temp_h, sizeof(prp_dma_pool), cudaMemcpyHostToDevice));
    }
    __host__ void s_queue_fini() {
        decltype(prp_dma_pool) prp_temp_h;
        cudaErrChk(cudaMemcpy(&prp_temp_h, &prp_dma_pool, sizeof(prp_dma_pool), cudaMemcpyDeviceToHost));
        prp_temp_h.prp_pool_fini_h();

        decltype(lq) lq_temp_h;
        cudaErrChk(cudaMemcpy(&lq_temp_h, &lq, sizeof(lq), cudaMemcpyDeviceToHost));
        lq_temp_h.lqueue_fini_h();
    }

    __forceinline__ __device__ sq_ptr_t find_poll_slot();
};

__forceinline__ __device__ sq_ptr_t
s_queue::find_poll_slot() {
    sq_ptr_t temp_idx;
    uint32_t lookout_val;

    // if constexpr (true) {
    if constexpr (false) {
        // uint32_t lookout = 512;
        // uint32_t lookout = 2048;
        uint32_t lookout = 4096; // this is good for case 2
        // uint32_t lookout = 8192;
        lookout_val = lq.nextpoll.value.fetch_add(1, cuda::memory_order_relaxed);
        lookout_val = lookout_val % lookout;
        temp_idx = lq.head.value.load(cuda::memory_order_relaxed) + lookout_val;
    } else {
        uint32_t lookout;
        uint64_t lookouts64;
        uint32_t lookout_mask;
        sq_ptr_t local_head;
        lookout_mask = __activemask();
        uint32_t lookout_leader = __ffs(lookout_mask) - 1;
        uint32_t lookout_total = __popc(lookout_mask);
        uint32_t lookout_lanemask_lt;
        asm("mov.u32 %0, %%lanemask_lt;"
            : "=r"(lookout_lanemask_lt));
        uint32_t lookout_rank = __popc(lookout_mask & lookout_lanemask_lt);
        if (lookout_rank == 0) {
            lookout_val = lq.nextpoll.value.fetch_add(lookout_total, cuda::memory_order_relaxed);

            local_head = lq.head.value.load(cuda::memory_order_relaxed);
            sq_ptr_t local_tail = lq.tail.value.load(cuda::memory_order_relaxed);
            uint32_t lookout_temp;
            if (local_tail <= local_head) {
                lookout_temp = 0;
            } else {
                lookout_temp = local_tail - local_head;
            }
            // lookout = lookout_temp + 8;
            lookout = lookout_temp + 64;
            force_assert(lookout < 65536);

            lookouts64 = (lookout << 16) | (0x0000FFFFU & lookout_val);
            lookouts64 = ((uint64_t)local_head << 32) | lookouts64;
        }
        lookouts64 = __shfl_sync(lookout_mask, lookouts64, lookout_leader);
        lookout = (uint32_t)((uint32_t)lookouts64 >> 16);
        lookout_val = (((uint32_t)lookouts64 & 0x0000FFFFU) + lookout_rank) % lookout;
        local_head = (uint32_t)(lookouts64 >> 32);
        temp_idx = local_head + lookout_val;
    }

    return temp_idx;
}

struct __align__(8) wait_queue_entry {
    uint64_t cid;
};

struct __align__(32) wait_queue {
    class lqueue<struct wait_queue_entry, wq_status_t, wq_ptr_t, n_wq_entries> lq;
    uint32_t my_idx;

    __device__ void wait_queue_init_d(uint32_t idx, uint32_t wq_thresh) {
        lq.lqueue_init_d(wq_thresh);
        my_idx = idx;
    }
    __device__ void wait_queue_fini_d() {
        printf("%s: waitq[%u]: %u %u\t", __func__, my_idx, lq.head.value.load(), lq.tail.value.load());
        if (my_idx % 4 == 1)
            printf("\n");
        lq.lqueue_fini_d();
    }

    __host__ void wait_queue_init_h(uint32_t n_queue_entries) {
        static_assert(std::is_standard_layout_v<wait_queue> == true);
        static_assert(std::is_trivially_copyable_v<wait_queue> == true);

        decltype(lq) lq_temp_h;
        lq_temp_h.lqueue_init_h();
        cudaErrChk(cudaMemcpy(&lq, &lq_temp_h, sizeof(lq), cudaMemcpyHostToDevice));
    }
    __host__ void wait_queue_fini_h() {
        decltype(lq) lq_temp_h;
        cudaErrChk(cudaMemcpy(&lq_temp_h, &lq, sizeof(lq), cudaMemcpyDeviceToHost));
        lq_temp_h.lqueue_fini_h();
    }

    __forceinline__ __device__ wq_ptr_t find_poll_slot();
};

__forceinline__ __device__ wq_ptr_t
wait_queue::find_poll_slot() {
    wq_ptr_t temp_idx;
    wq_ptr_t local_wq_head = lq.head.value.load(cuda::memory_order_relaxed);
    wq_ptr_t local_wq_tail = lq.tail.value.load(cuda::memory_order_relaxed);
    uint32_t lookout = local_wq_tail - local_wq_head;
    lookout += 128;

    // wq_ptr_t lookout_add;
    // uint32_t lookout_mask = __activemask();
    // uint32_t lookout_leader = __ffs(lookout_mask) - 1;
    // uint32_t lookout_total = __popc(lookout_mask);
    // uint32_t lookout_lanemask_lt;
    // asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lookout_lanemask_lt));
    // uint32_t lookout_rank = __popc(lookout_mask & lookout_lanemask_lt);
    // if (lookout_rank == 0) {
    //     lookout_add = local_waitq->lookout_assign.fetch_add(lookout_total, cuda::memory_order_relaxed);
    // }
    // lookout_add = __shfl_sync(lookout_mask, lookout_add, lookout_leader) + lookout_rank;
    // wq_ptr_t my_cb_inc = local_wq_head + lookout_add;

    wq_ptr_t lookout_add = lq.nextpoll.value.fetch_add(1, cuda::memory_order_relaxed) % lookout;
    temp_idx = lq.head.value.load(cuda::memory_order_relaxed) + lookout_add;

    return temp_idx;
}

struct __align__(32) wait_ds {
    // Wait Queues
    struct wait_queue __align__(128) * wait_queue;
    uint32_t num_lib_waitq;

    cuda::atomic<int32_t, cuda::thread_scope_device> *w_queue_poller;

    uint32_t check_func_caller_tid;
    cuda::atomic<int64_t, cuda::thread_scope_device> check_func_return;

    __device__ void wait_ds_init_d(uint32_t wq_thresh) {
        for (uint32_t i = 0; i < num_lib_waitq; ++i) {
            new (&(wait_queue[i])) struct wait_queue;
            wait_queue[i].wait_queue_init_d(i, wq_thresh);
            w_queue_poller[i].store(-1);
        }
        check_func_caller_tid = 0;
    }
    __device__ void wait_ds_fini_d() {
        uint64_t hsum = 0;
        uint64_t tsum = 0;
        for (uint32_t i = 0; i < num_lib_waitq; ++i) {
            hsum += wait_queue[i].lq.head.value.load();
            tsum += wait_queue[i].lq.tail.value.load();
            wait_queue[i].wait_queue_fini_d();
            wait_queue[i].~wait_queue();
        }
        printf("%s: hsum: %lu, tsum: %lu\n", __func__, hsum, tsum);
    }

    __host__ void wait_ds_init_h(uint32_t n_lib_wq, uint32_t n_lib_wq_entries) {
        static_assert(std::is_standard_layout_v<wait_ds> == true);
        static_assert(std::is_trivially_copyable_v<wait_ds> == true);

        std::vector<struct wait_queue> wait_queue_temp_h(n_lib_wq);
        for (uint32_t i = 0; i < n_lib_wq; ++i) {
            wait_queue_temp_h[i].wait_queue_init_h(n_wq_entries);
        }
        cudaErrChk(cudaMalloc(&wait_queue, sizeof(*wait_queue) * n_lib_wq));
        cudaErrChk(cudaMemcpy(wait_queue, wait_queue_temp_h.data(), sizeof(*wait_queue) * n_lib_wq, cudaMemcpyHostToDevice));

        cudaErrChk(cudaMalloc(&w_queue_poller, sizeof(cuda::atomic<int32_t, cuda::thread_scope_device>) * n_lib_wq));

        num_lib_waitq = n_lib_wq;
    }
    __host__ void wait_ds_fini_h() {
        cudaErrChk(cudaFree(w_queue_poller));

        std::vector<struct wait_queue> wait_queue_temp_h(num_lib_waitq);
        cudaErrChk(cudaMemcpy(wait_queue_temp_h.data(), wait_queue, sizeof(*wait_queue) * num_lib_waitq, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < num_lib_waitq; ++i) {
            wait_queue_temp_h[i].wait_queue_fini_h();
        }
        cudaErrChk(cudaFree(wait_queue));
    }

    __forceinline__ __device__ int64_t
    _gpu_wait_any_tag(const uint32_t wq_id, const uint32_t tag);
};

// Wait for a limited bound, then quickly exit if entry is not valid
__forceinline__ __device__ int64_t
wait_ds::_gpu_wait_any_tag(const uint32_t wq_id, const uint32_t tag) {
    struct wait_queue *local_waitq = &(wait_queue[wq_id]);

    // wq_ptr_t local_wq_head;
    // wq_ptr_t local_wq_tail;
    // uint32_t lookout_mask = __activemask();
    // uint32_t lookout_leader = __ffs(lookout_mask) - 1;
    // uint32_t lookout_total = __popc(lookout_mask);
    // uint32_t lookout_lanemask_lt;
    // asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lookout_lanemask_lt));
    // uint32_t lookout_rank = __popc(lookout_mask & lookout_lanemask_lt);
    // if (lookout_rank == 0) {
    //     local_wq_head = local_waitq->lq.head.value.load(cuda::memory_order_relaxed);
    //     local_wq_tail = local_waitq->lq.tail.value.load(cuda::memory_order_relaxed);
    // }
    // local_wq_head = __shfl_sync(lookout_mask, local_wq_head, lookout_leader);
    // local_wq_tail = __shfl_sync(lookout_mask, local_wq_tail, lookout_leader);

    wq_ptr_t local_wq_head = local_waitq->lq.head.value.load(cuda::memory_order_relaxed);
    wq_ptr_t local_wq_tail = local_waitq->lq.tail.value.load(cuda::memory_order_relaxed);
    if (local_wq_tail <= local_wq_head) {
        return GPU_WAIT_NOT_FOUND;
    }

    uint32_t wait_cnt = 0;

redo:
    wq_ptr_t temp_idx = local_waitq->find_poll_slot();
    const uint32_t my_idx = temp_idx % n_wq_entries;

    cuda::atomic<wq_status_t, cuda::thread_scope_device> *local_waitq_entry_status = &(local_waitq->lq.entry_status[my_idx].value);

    wq_status_t status_load;
    wq_status_t temp;
    uint32_t ns = 8;
    while (true) {
        if (wait_cnt > 4) {
            return GPU_WAIT_NOT_FOUND;
        }
        ++wait_cnt;
        status_load = local_waitq_entry_status->load(cuda::memory_order_relaxed);
        if ((tag != 0) && ((status_load >> 24) != tag))
            goto redo;
        temp = status_load & 0x0000FFFFU;
        if (temp == ENTRY_VALID)
            break;
        if (temp == ENTRY_PROCESSED || temp == ENTRY_FILLING) {
            return GPU_WAIT_NOT_FOUND;
        }
        __nanosleep(ns);
    }

    int64_t cid = GPU_WAIT_NOT_FOUND;
    if (local_waitq_entry_status->compare_exchange_strong(status_load,
                                                          (wq_status_t)ENTRY_FILLING, cuda::memory_order_acquire) == true) {
        cid = local_waitq->lq.entry_buffer[my_idx].cid;

        local_waitq_entry_status->store((wq_status_t)ENTRY_PROCESSED, cuda::memory_order_release);
    }
    return cid;
}

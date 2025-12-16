#pragma once

#include "cuda.h"
#include <cuda/atomic>

#include "macro.h"

template <typename T>
struct __align__(32) lqueue_status {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[32];
    };
};

template <typename T>
struct __align__(8) lqueue_ptr {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[4];
    };
};

template <typename qentry_t, typename qstatus_t, typename qptr_t, uint32_t qlen>
class __align__(128) lqueue {
    public:
    struct lqueue_status<qstatus_t> __align__(32) * entry_status;
    qentry_t *entry_buffer;
    struct lqueue_ptr<qptr_t> head;
    struct lqueue_ptr<qptr_t> tail;
    struct lqueue_ptr<qptr_t> nextpoll;

    uint32_t threshold;
    uint32_t num_increment_skip;

    __device__ void lqueue_init_d(uint32_t thresh) {
        head.value.store(0, cuda::memory_order_relaxed);
        tail.value.store(0, cuda::memory_order_relaxed);
        nextpoll.value.store(0, cuda::memory_order_relaxed);

        threshold = thresh;
        num_increment_skip = 0;
    }
    __device__ void lqueue_fini_d() {
    }

    __host__ void lqueue_init_h() {
        static_assert(std::is_standard_layout_v<lqueue> == true);
        static_assert(std::is_trivially_copyable_v<lqueue> == true);

        std::vector<struct lqueue_status<qstatus_t>> temp1_h(qlen);
        for (size_t i = 0; i < temp1_h.size(); ++i) {
            temp1_h[i].value = ENTRY_EMPTY;
        }

        std::remove_pointer_t<decltype(this)> lq;

        cudaErrChk(cudaMalloc(&(lq.entry_status), sizeof(*lq.entry_status) * qlen));
        cudaErrChk(cudaMemcpy(lq.entry_status, temp1_h.data(), sizeof(*lq.entry_status) * qlen, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMalloc(&(lq.entry_buffer), sizeof(*lq.entry_buffer) * qlen));
        cudaErrChk(cudaMemset(lq.entry_buffer, 0, sizeof(*lq.entry_buffer) * qlen));

        memcpy(this, &lq, sizeof(*this));
    }
    __host__ void lqueue_fini_h() {
        std::remove_pointer_t<decltype(this)> lq;
        memcpy(reinterpret_cast<char *>(&lq), this, sizeof(*this));

        cudaErrChk(cudaFree(lq.entry_buffer));
        cudaErrChk(cudaFree(lq.entry_status));
    }

    __device__ void head_increment();
    __device__ uint32_t get_n_pending();
    __device__ bool enqueue_getslot_simple(qptr_t *q_idx);
    __device__ bool enqueue_getslot(qptr_t *q_idxes, uint32_t n_slots);
};

template <typename qentry_t, typename qstatus_t, typename qptr_t, uint32_t qlen>
__forceinline__ __device__ void
lqueue<qentry_t, qstatus_t, qptr_t, qlen>::head_increment() {
    LIB_TID;
    const uint32_t lid = tid % 32;
    // constexpr uint32_t max_increment = 1024; // multiple of 32
    constexpr uint32_t max_increment = 4096; // multiple of 32

    qptr_t local_head;
    // qptr_t local_temp_head;
    if (lid == 0) {
        local_head = head.value.load(cuda::memory_order_relaxed);
        // local_temp_head = temp_head.value.load(cuda::memory_order_relaxed);
    }
    local_head = __shfl_sync(0xFFFFFFFF, local_head, 0);
    // local_temp_head = __shfl_sync(0xFFFFFFFF, local_temp_head, 0);

    qptr_t idx;
    uint32_t increment = 0;
    uint32_t break_now;
    // if (local_temp_head - local_head >= 32) {
    // batch check
    break_now = 0;
    for (increment = lid; increment < max_increment; increment += 32) {
        idx = (local_head + increment) % qlen;
        qstatus_t status = entry_status[idx].value.load(cuda::memory_order_relaxed);
        if (status != (qstatus_t)ENTRY_PROCESSED) {
            break_now = 1;
        }
        break_now = __any_sync(0xFFFFFFFF, break_now);
        if (break_now != 0) {
            break;
        }
        entry_status[idx].value.store((qstatus_t)ENTRY_EMPTY, cuda::memory_order_relaxed);
    }
    increment = __reduce_min_sync(0xFFFFFFFF, increment);
    // }

    if (increment == 0) {
        // single check
        if (lid == 0) {
            for (; increment < 64; increment += 1) {
                idx = (local_head + increment) % qlen;
                qstatus_t status = entry_status[idx].value.load(cuda::memory_order_relaxed);
                if (status != (qstatus_t)ENTRY_PROCESSED) {
                    break;
                }
                entry_status[idx].value.store(ENTRY_EMPTY, cuda::memory_order_relaxed);
            }
        }
        increment = __shfl_sync(0xFFFFFFFF, increment, 0);
    }

    if (increment == 0) {
        return;
    }

    if (lid == 0) {
        head.value.store(local_head + increment, cuda::memory_order_relaxed);
    }
    return;
}

template <typename qentry_t, typename qstatus_t, typename qptr_t, uint32_t qlen>
__forceinline__ __device__ uint32_t
lqueue<qentry_t, qstatus_t, qptr_t, qlen>::get_n_pending() {
    qptr_t local_head = head.value.load(cuda::memory_order_relaxed);
    qptr_t local_tail = tail.value.load(cuda::memory_order_relaxed);
    if (local_tail <= local_head) {
        return (uint32_t)0;
    }
    return (uint32_t)(local_tail - local_head);
}

template <typename qentry_t, typename qstatus_t, typename qptr_t, uint32_t qlen>
__forceinline__ __device__ bool
lqueue<qentry_t, qstatus_t, qptr_t, qlen>::enqueue_getslot_simple(qptr_t *q_idx) {
    uint32_t n_pending = get_n_pending();
    if (n_pending >= threshold)
        return false;

    *q_idx = tail.value.fetch_add(1, cuda::memory_order_relaxed) % qlen;

    qstatus_t expected = ENTRY_EMPTY;
    qstatus_t exchange_ret = entry_status[*q_idx].value.exchange(0x65734455, cuda::memory_order_relaxed);
    force_assert_printf(exchange_ret == expected, "%s: invalid status: %X\n", __func__, exchange_ret);

    return true;
}

template <typename qentry_t, typename qstatus_t, typename qptr_t, uint32_t qlen>
__forceinline__ __device__ bool
lqueue<qentry_t, qstatus_t, qptr_t, qlen>::enqueue_getslot(qptr_t *q_idxes, uint32_t n_slots) {
    uint32_t n_pending = get_n_pending();
    if (n_pending >= threshold)
        return false;

    qptr_t local_tail;
    if (n_slots == 1) {
        uint32_t tail_mask = __activemask();
        uint32_t tail_leader = __ffs(tail_mask) - 1;
        uint32_t tail_total = __popc(tail_mask);
        uint32_t tail_lanemask_lt;
        asm("mov.u32 %0, %%lanemask_lt;"
            : "=r"(tail_lanemask_lt));
        uint32_t tail_rank = __popc(tail_mask & tail_lanemask_lt);
        if (tail_rank == 0) {
            local_tail = tail.value.fetch_add(tail_total, cuda::memory_order_relaxed);
        }
        local_tail = __shfl_sync(tail_mask, local_tail, tail_leader) + tail_rank;
    } else {
        local_tail = tail.value.fetch_add(n_slots, cuda::memory_order_relaxed);
    }

    for (uint32_t i = 0; i < n_slots; ++i) {
        q_idxes[i] = (local_tail + i) % qlen;
    }

    qstatus_t expected = ENTRY_EMPTY;
    qstatus_t exchange_ret = entry_status[q_idxes[n_slots - 1]].value.exchange(0x65734455, cuda::memory_order_relaxed);
    force_assert_printf(exchange_ret == expected, "%s: invalid status: %X\n", __func__, exchange_ret);

    for (uint32_t i = 0; i < n_slots - 1; ++i) {
        qstatus_t s = entry_status[q_idxes[i]].value.load(cuda::memory_order_relaxed);
        force_assert_printf(s == (qstatus_t)ENTRY_EMPTY, "%s: invalid status: %X\n", __func__, s);
    }
    return true;
}

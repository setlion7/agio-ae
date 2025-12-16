#pragma once

#include <cstdio>
#include <cstdint>
#include <cuda/atomic>
#include <cuda.h>

#include "ds.cuh"
#include "iocmd.cuh"
#include "macro.h"

__forceinline__ __device__ void
_notify_io(class gio *gp, const uint32_t qid, const uint64_t cid, const bool notify_wq) {
    if (notify_wq) {
        // Post to WQ
        struct wait_queue *local_waitq = &(gp->wait_ds.wait_queue[qid]);

        uint32_t wait_idx;
        while (true) {
            bool ret = local_waitq->lq.enqueue_getslot_simple(&wait_idx);
            if (ret)
                break;
            __nanosleep(32);
        }
        struct lqueue_status<wq_status_t> *wq_entry_status = &(local_waitq->lq.entry_status[wait_idx]);

        local_waitq->lq.entry_buffer[wait_idx].cid = cid;
        wq_entry_status->value.store((wq_status_t)ENTRY_VALID, cuda::memory_order_release);
    } else {
        uint32_t stage_id = (simple_hash_len4((uint8_t *)&cid)) % gp->n_stage;
        class lstage *sp = &(gp->stage[stage_id]);
        sp->post_entry(cid);
    }
}

template <bool is_write>
__forceinline__ __device__ void
_gpu_io_aligned(class gio *gp, const uint32_t sq_id, const uint64_t cid, const uint32_t devid,
                 const uint64_t io_addr, const uint64_t start_byte, const uint64_t num_bytes, const bool notify_wq) {
    USER_GID;
    force_assert(start_byte % G_LBA_SIZE == 0);
    force_assert(num_bytes % G_LBA_SIZE == 0);

    struct s_queue *local_sq = &(gp->s_queue[sq_id]);
    struct prp_pool *prps = &(local_sq->prp_dma_pool);
    class Controller *ctrl_ptr = gp->ctrl_dptrs[devid];
    const uint32_t nvme_qp_num = gid % ctrl_ptr->n_qps;

    io_proc_data<is_write>(gp, prps, ctrl_ptr, nvme_qp_num, io_addr, start_byte / 512, num_bytes / 512);

    _notify_io(gp, sq_id, cid, notify_wq);
}

template <bool is_write>
__forceinline__ __device__ void
_gpu_io_via_cache(class gio *gp, const uint32_t lib_qid, const uint64_t cid, const uint32_t devid,
                   void *dma_buf_vaddr, const uint64_t start_byte, const uint64_t num_bytes, const bool notify_wq) {
    USER_GID;

    cache_copy_data<is_write>(gp, devid, gid % gp->ctrl_dptrs[devid]->n_qps, gp->cp->b_arr, dma_buf_vaddr, start_byte, num_bytes);

    _notify_io(gp, lib_qid, cid, notify_wq);
}

// 0: issued successfully
// 1: no-op
// -1: retry later
__forceinline__ __device__ int32_t
_gpu_submit(class gio *gptr, const uint32_t qid, struct iocb *cb, uint32_t len) {
    for (uint32_t i = 0; i < len; ++i) {
        if (cb[i].valid == false || (cb[i].submitted == true))
            return 1;
    }

    struct s_queue *local_s_queue = &(gptr->s_queue[qid]);
    uint32_t issue_pending = local_s_queue->current_n_items.fetch_add(1, cuda::memory_order_relaxed);
    // uint32_t wait_pending = gpu_num_wait_pending(gptr, qid);
    // if (issue_pending < gptr->sq_threshold && wait_pending < gptr->wq_threshold) {
    if (issue_pending < gptr->sq_threshold) {

        for (uint32_t i = 0; i < len; ++i) {
            uint64_t addr;
            GPU_SUBMIT_MEMALLOC(gptr->mptr, cb[i].opcode.opcode_id,
                                addr, cb[i].address, cb[i].num_bytes);
            if (addr == 0) {
                force_assert_printf(false, "memalloc failed in lib, n_bytes: %lu\n", cb[i].num_bytes);
                // redo = true;
            }

            const uint64_t command_id = cb[i].command_id;
            const uint64_t l_start_byte = cb[i].start_byte;
            const uint64_t l_num_bytes = cb[i].num_bytes;
            const uint32_t nvme_device_id = cb[i].opcode.device_id;
            const uint32_t l_opcode = cb[i].opcode.opcode_id;

            if (l_opcode == GPU_OPCODE_READ_DIRECT) {
                _gpu_io_aligned<false>(gptr, qid, command_id, nvme_device_id, addr, l_start_byte, l_num_bytes, cb[i].tag);
            } else if (l_opcode == GPU_OPCODE_READ_CACHE) {
                _gpu_io_via_cache<false>(gptr, qid, command_id, nvme_device_id, (void *)addr, l_start_byte, l_num_bytes, cb[i].tag);
            } else if (l_opcode == GPU_OPCODE_WRITE_DIRECT) {
                _gpu_io_aligned<true>(gptr, qid, command_id, nvme_device_id, addr, l_start_byte, l_num_bytes, cb[i].tag);
            } else if (l_opcode == GPU_OPCODE_WRITE_CACHE) {
                _gpu_io_via_cache<true>(gptr, qid, command_id, nvme_device_id, (void *)addr, l_start_byte, l_num_bytes, cb[i].tag);
            } else {
                force_assert(false);
            }

            cb[i].submitted = true;
        }

        local_s_queue->current_n_items.fetch_sub(1, cuda::memory_order_relaxed);
        return 0;
    }
    local_s_queue->current_n_items.fetch_sub(1, cuda::memory_order_relaxed);
    return -1;
}

// 0: issued successfully, set cb.submitted to true
// 1: no-op
// -1: retry later
__forceinline__ __device__ int32_t
_gpu_submit_async(class gio *gp, const uint32_t qid, class iocb *cb, uint32_t len) {
    // Check in upper layer
    // for (uint32_t i = 0; i < len; ++i) {
    //     if (cb[i].valid == false || (cb[i].submitted == true))
    //         return 1;
    // }

    sq_ptr_t local_sq_idx[8];
    uint64_t addr[8];
    force_assert(len <= 8);

    struct s_queue *local_s_queue = &(gp->s_queue[qid]);

    bool enqueue_ret = local_s_queue->lq.enqueue_getslot(local_sq_idx, len);
    if (enqueue_ret) {
        bool redo = false;
        for (uint32_t i = 0; i < len; ++i) {
            if (cb[i].opcode.opcode_id == GPU_OPCODE_NOOP) {
                continue;
            }
            GPU_SUBMIT_MEMALLOC(gp->mptr, cb[i].opcode.opcode_id,
                                addr[i], cb[i].address, cb[i].num_bytes);
            if (addr[i] == 0) {
                // force_assert_printf(false, "memalloc failed in interface, n_bytes: %lu\n", cb[i].num_bytes);
                redo = true;
            }
        }
        if (redo) {
            for (uint32_t i = 0; i < len; ++i) {
                local_s_queue->lq.entry_status[local_sq_idx[i]].value.store((sq_status_t)(ENTRY_NOOP), cuda::memory_order_release);
            }
            return -1;
        }

        for (uint32_t i = 0; i < len; ++i) {
            struct io_s_entry *selected_entry = &(local_s_queue->lq.entry_buffer[local_sq_idx[i]]);

            if (cb[i].opcode.opcode_id != GPU_OPCODE_NOOP) {
                selected_entry->address = addr[i];
                selected_entry->start_byte = cb[i].start_byte;
                selected_entry->num_bytes = cb[i].num_bytes;
            }
            selected_entry->command_id = cb[i].command_id;
            selected_entry->opcode = cb[i].opcode;

            // cb[i].address = addr;
            // my_memcpy(selected_entry, &cb[i], sizeof(struct io_s_entry));

            // cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
        }

        for (uint32_t i = 0; i < len; ++i) {
            const uint32_t tag_shift = cb[i].tag << 24;
            if (tag_shift) {
            }
            const uint32_t vec = i << (20) | (len << 16);
            local_s_queue->lq.entry_status[local_sq_idx[i]].value.store((sq_status_t)(tag_shift | vec | ENTRY_VALID), cuda::memory_order_release);
            cb[i].submitted = true;
        }

        return 0;
    }
    // local_s_queue->n_thresh_exceeded.fetch_add(1, cuda::memory_order_relaxed);
    return -1;
}

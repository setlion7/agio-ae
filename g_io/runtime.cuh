#pragma once

#include <cuda/atomic>

#include "cuda.h"

#include "helper_headers/fassert.cuh"
#include "helper_headers/gpuHash.cuh"

#include "ds.cuh"
#include "comm.cuh"
#include "iocmd.cuh"
#include "macro.h"

__forceinline__ __device__ void
gio::runtime_main(const uint32_t tid, const uint32_t groupId) {
    const uint32_t lid = tid & 0x1f;
    const uint32_t wid = tid / 32;
    uint32_t smid = get_smid2();
    uint32_t smid_fromzero;

    // launch check
    if (lid == 0) {
        rt_start_num_warps.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncwarp();
    {
        uint32_t ns = 2;
        uint32_t wait_cnt = 0;
        while (rt_start_num_warps.load(cuda::memory_order_relaxed) != total_num_lib_warps) {
            __nanosleep(ns);
            if (ns < 1024)
                ns *= 2;
            force_assert_printf(wait_cnt < 1024 * 1024 * 16, "%s: expected %u, but current: %u\n",
                                __func__, rt_start_num_warps.load(cuda::memory_order_relaxed), total_num_lib_warps);
        }
        __syncwarp();
    }

    // smid
    if (lid == 0 && (threadIdx.x / 32 == 0)) {
        int32_t sm_expected = -1;
        if (smid_table[smid].compare_exchange_strong(sm_expected, -2, cuda::memory_order_relaxed) == true) {
            smid_fromzero = smid_assign.fetch_add(1, cuda::memory_order_relaxed);
            smid_table[smid].store(smid_fromzero, cuda::memory_order_relaxed);
            // printf("%s: sm selected warp, threadIdx.x: %u, wid: %u, smid: %u, smid_fromzero: %u\n",
            //        __func__, threadIdx.x, wid, smid, smid_fromzero);
        }
    }
    __syncwarp();

    while (true) {
        int32_t temp = smid_table[smid].load(cuda::memory_order_relaxed);
        if (temp >= 0) {
            smid_fromzero = (uint32_t)temp;
            break;
        }
    }

    if (tid == 0) {
        rt_running = 1;
        printf("%s: running...\n", __func__);
    }
    __syncwarp();

    const uint32_t local_sq_num = smid_fromzero * n_queues_per_rtsm + ((threadIdx.x / 32) % n_queues_per_rtsm);
    struct s_queue *local_sq = &(s_queue[local_sq_num]);

    constexpr uint32_t n_sm_per_poller_warp = 2;
    uint32_t n_queues_per_poller_warp = n_queues_per_rtsm * n_sm_per_poller_warp;

    int32_t poll_sqid_start = 0;
    uint32_t sq_poller = 0;
    if (lid == 0 && (threadIdx.x / 32 == 0)) {
        poll_sqid_start = smid_fromzero * n_queues_per_poller_warp;
        int32_t expected = -1;
        if (s_queue_poller[poll_sqid_start].compare_exchange_strong(expected, wid, cuda::memory_order_relaxed) == true) {
            sq_poller = 1;
        }
    }
    poll_sqid_start = __shfl_sync(0xFFFFFFFF, poll_sqid_start, 0);
    sq_poller = __shfl_sync(0xFFFFFFFF, sq_poller, 0);

    if (sq_poller == 1) {
        __syncwarp();
        uint32_t check_acc = 0;

        if (lid == 0) {
            // printf("%s: tid: %u, threadIdx.x: %u, wid: %u, groupId: %u, smid: %u, smid_fromzero: %u,"
            //        " sqnum: %u, n_sq: %u, poll_sqid_start: %u, poller start\n",
            //        __func__, tid, threadIdx.x, wid, groupId, smid, smid_fromzero,
            //        local_sq_num, n_sq, poll_sqid_start);
        }

        while (true) {
            for (uint32_t i = 0; i < n_queues_per_poller_warp; ++i) {
                s_queue[poll_sqid_start + i].lq.head_increment();
                wait_ds.wait_queue[poll_sqid_start + i].lq.head_increment();
            }

            if (smid_fromzero == 0) {
                dy.try_update();
            }

            ++check_acc;
            if (check_acc > 1024) {
                if (local_sq->qstate.load(cuda::memory_order_relaxed) == 0) {
                    break;
                }
                check_acc = 0;
            }
        }
    }
    __syncwarp();

    while (local_sq->qstate.load(cuda::memory_order_relaxed) != 0) {
        uint32_t ns1 = 32;
        uint32_t wait_cnt1 = 0;
        sq_ptr_t temp_idx;
    redo_lookout:

        temp_idx = local_sq->find_poll_slot();

        const sq_ptr_t sq_entry_idx = temp_idx % n_sq_entries;
        cuda::atomic<sq_status_t, cuda::thread_scope_device> *sq_entry_status =
            &(local_sq->lq.entry_status[sq_entry_idx].value);
        sq_status_t sqs_temp;
        sq_status_t sqs_temp_status;
        uint32_t packet_idx;
        uint32_t status_check_cnt = 0;
        while (true) {
            sqs_temp = sq_entry_status->load(cuda::memory_order_relaxed);
            sqs_temp_status = sqs_temp & 0x0000FFFFU;
            if (sqs_temp_status == (sq_status_t)ENTRY_VALID) {
                break;
            } else if (sqs_temp_status == (sq_status_t)ENTRY_PROCESSING) {
                goto redo_lookout;
            } else if (sqs_temp_status == (sq_status_t)ENTRY_NOOP) {
                // sq_entry_status->store((sq_status_t)ENTRY_PROCESSED, cuda::memory_order_release);
                sq_entry_status->compare_exchange_weak(sqs_temp, (sq_status_t)ENTRY_PROCESSED, cuda::memory_order_release);
                goto redo_lookout;
            }
            if (status_check_cnt > 32) {
                if (local_sq->qstate.load(cuda::memory_order_relaxed) == 0) {
                    goto loop_exit;
                }
                goto redo_lookout;
            }
            status_check_cnt++;
            sleep_wait_assert_printf(ns1, 128, wait_cnt1, 1024 * 1024 * 256U,
                                     "sq entry not valid, sqid: %u, idx: %u\n", local_sq_num, sq_entry_idx);
        }

        sq_status_t status_store = ENTRY_PROCESSING;
        bool sqs_exchange_ret = sq_entry_status->compare_exchange_weak(sqs_temp, status_store, cuda::memory_order_acquire);
        if (sqs_exchange_ret == false) {
            local_sq->n_cmp_false.fetch_add(1, cuda::memory_order_relaxed);
            continue;
        }

        packet_idx = (sqs_temp >> 20) & 0x000000FU;
        const uint32_t packet_len = (sqs_temp >> 16) & 0x0000000F;
        gpua_tag_t tag = sqs_temp >> 24;

        process_io(local_sq, (temp_idx) % n_sq_entries, tag);

        if (packet_idx == 0 && (packet_len != 1)) {
            for (uint32_t i = 1; i < packet_len; ++i) {
                while (true) {
                    sq_status_t others = local_sq->lq.entry_status[(temp_idx + i) % n_sq_entries].value.load(cuda::memory_order_relaxed);
                    if (others == ENTRY_PROCESSED)
                        break;
                    __nanosleep(8);
                }
            }
        }

        sq_entry_status->store((sq_status_t)ENTRY_PROCESSED, cuda::memory_order_release);
        notify_io(local_sq, (temp_idx) % n_sq_entries, tag);
    } // while

loop_exit:
    int temp;
    (void)temp;
}

__forceinline__ __device__ void
gio::process_io(struct s_queue *local_sq, sq_ptr_t sq_entry_idx, gpua_tag_t tag) {
    LIB_GID;
    struct io_s_entry *sq_entry_buffer = &(local_sq->lq.entry_buffer[sq_entry_idx]);
    cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
    struct prp_pool *prps = &(local_sq->prp_dma_pool);
    const uint64_t l_address = sq_entry_buffer->address;
    const uint64_t l_start_byte = sq_entry_buffer->start_byte;
    const uint64_t l_num_bytes = sq_entry_buffer->num_bytes;
    const uint32_t nvme_device_id = sq_entry_buffer->opcode.device_id;
    const uint32_t l_opcode = sq_entry_buffer->opcode.opcode_id;
    class Controller *ctrl_ptr = ctrl_dptrs[nvme_device_id];
    const uint32_t nvme_qp_num = gid % ctrl_ptr->n_qps;

    if (l_opcode == GPU_OPCODE_NOOP) {
        // Do nothing

    } else if (l_opcode == GPU_OPCODE_READ_DIRECT) {
        io_proc_data<false>(this, prps, ctrl_ptr, nvme_qp_num, l_address, l_start_byte / 512, l_num_bytes / 512);

    } else if (l_opcode == GPU_OPCODE_READ_CACHE) {
        cache_copy_data<false>(this, nvme_device_id, nvme_qp_num, cp->b_arr, (void *)l_address, l_start_byte, l_num_bytes);

    } else if (l_opcode == GPU_OPCODE_WRITE_DIRECT) {
        io_proc_data<true>(this, prps, ctrl_ptr, nvme_qp_num, l_address, l_start_byte / 512, l_num_bytes / 512);

    } else if (l_opcode == GPU_OPCODE_WRITE_CACHE) {
        cache_copy_data<true>(this, nvme_device_id, nvme_qp_num, cp->b_arr, (void *)l_address, l_start_byte, l_num_bytes);

    } else {
        force_assert_printf(false, "%s: lopcode: %u, sqidx: %u\n", __func__, l_opcode, sq_entry_idx);
    }
}

__forceinline__ __device__ void
gio::notify_io(struct s_queue *local_sq, sq_ptr_t sq_entry_idx, gpua_tag_t tag) {
    LIB_GID;
    struct io_s_entry *sq_entry_buffer = &(local_sq->lq.entry_buffer[sq_entry_idx]);
    const uint64_t command_id = sq_entry_buffer->command_id;

    if (tag & tag_enabled) {
        // Some requests may not require callbacks
        const uint32_t local_waitq_num = gid % wait_ds.num_lib_waitq;
        struct wait_queue *local_waitq = &(wait_ds.wait_queue[local_waitq_num]);

        uint32_t wait_idx;
        while (true) {
            // bool ret = local_waitq->lq.enqueue_getslot(&wait_idx, 1);
            bool ret = local_waitq->lq.enqueue_getslot_simple(&wait_idx);
            if (ret)
                break;
            __nanosleep(32);
        }
        cuda::atomic<wq_status_t, cuda::thread_scope_device> *waitq_entry_status = &(local_waitq->lq.entry_status[wait_idx].value);

        local_waitq->lq.entry_buffer[wait_idx].cid = command_id;
        waitq_entry_status->store((wq_status_t)((tag << 24) | ENTRY_VALID), cuda::memory_order_release);
    } else {
        uint32_t stage_id = (simple_hash_len4((uint8_t *)&command_id)) % n_stage;
        class lstage *sp = &(stage[stage_id]);
        sp->post_entry(command_id);
    }
}

__forceinline__ __device__ void
gio::shutdown() {
    LIB_TID;

    if (tid == 0) {
        for (uint32_t i = 0; i < n_sq; ++i) {
            s_queue[i].qstate.store(0, cuda::memory_order_relaxed);
        }
    }
}

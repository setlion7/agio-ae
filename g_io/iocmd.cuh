#pragma once

#include <cuda.h>

#include "ctrl.h"
#include "macro.h"
#include "nvm_cmd.h"
#include "nvm_parallel_queue.h"
#include "nvm_queue.h"
#include "queue.h"

#include "helper_headers/gpuUtil.cuh"

#include "ds.cuh"

// Build PRP and return prp page number if prp_pool is used
// If prp_pool is NOT used, return regative number
__forceinline__ __device__ int32_t
nvme_build_prp(class gio *gp, struct prp_pool *prps,
               uint64_t start_ioaddr, uint64_t end_ioaddr, uint64_t *prp1, uint64_t *prp2) {
    *prp1 = start_ioaddr;
    int32_t additional_dma_page_num = -1;

    uint32_t cross_page_boundary_count = (end_ioaddr >> 12) - (start_ioaddr >> 12);
    if (cross_page_boundary_count == 0) {
        *prp2 = 0;
    } else if (cross_page_boundary_count == 1) {
        // Crosses one memory page boundary
        *prp2 = (*prp1 + 4096) & 0xFFFFFFFFFFFFF000ULL; // Offset must be zero for prp2
    } else if (cross_page_boundary_count <= 512) {
        // Build PRP list
        uint32_t local_next_dma = prps->prp_next_dma_page.value.fetch_add(1, cuda::memory_order_relaxed);
        additional_dma_page_num = local_next_dma % prps->prp_dma_num_pages;
        force_assert(prps->prp_dma_status[additional_dma_page_num].value.load(cuda::memory_order_relaxed) == ENTRY_EMPTY);
        prps->prp_dma_status[additional_dma_page_num].value.store(ENTRY_VALID, cuda::memory_order_relaxed);

        // printf("%s: dma_ioaddr: %lX, end_ioaddr: %lX, cross_page_boundary_count: %u, num_lba_blocks: %hu\n",
        //        __func__, dma_ioaddr, end_ioaddr, cross_page_boundary_count, num_lba_blocks);

        uint64_t *prp_list_ptr = (uint64_t *)((char *)prps->prp_dma_buffer_dptr + (4096 * additional_dma_page_num));
        for (uint32_t i = 0; i < cross_page_boundary_count; ++i) {
            prp_list_ptr[i] = (*prp1 + (4096 * (i + 1))) & 0xFFFFFFFFFFFFF000ULL;
        }
        *prp2 = prps->prp_dma_buffer_ioaddr + (4096 * additional_dma_page_num);
    } else {
        force_assert_printf(false, "nvme size too large, start_ioaddr: %lX, end_ioaddr: %lX, page count: %u\n",
                            start_ioaddr, end_ioaddr, cross_page_boundary_count);
    }
    return additional_dma_page_num;
}

__forceinline__ __device__ void
nvm_cmd_process(class gio *gp, struct prp_pool *prps,
                QueuePair *qp, const uint8_t opcode,
                const uint64_t dma_ioaddr, const uint64_t lba_start, const uint16_t num_lba_blocks) {
    const uint16_t cid = get_cid(&(qp->sq));
    uint64_t prp1;
    uint64_t prp2;
    uint64_t end_ioaddr = dma_ioaddr + ((uint64_t)num_lba_blocks * 512ULL) - 1;
    int32_t prp_dma_used = nvme_build_prp(gp, prps, dma_ioaddr, end_ioaddr, &prp1, &prp2);

    // printf("%s: dma_ioaddr: %lX, cid: %hu, lba_start: %lu, num_lba_blocks: %hu\n",
    //        __func__, dma_ioaddr, cid, lba_start, num_lba_blocks);

    nvm_cmd_t cmd;
    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, lba_start, num_lba_blocks);

    sq_enqueue(&qp->sq, &cmd);

    uint32_t head, head_;
    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

    qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

    if (prp_dma_used >= 0) {
        prps->prp_dma_status[prp_dma_used].value.store(ENTRY_EMPTY, cuda::memory_order_relaxed);
    }
    put_cid(&(qp->sq), cid);
}

template <bool is_write>
__forceinline__ __device__ void
io_proc_data(class gio *gp, struct prp_pool *prps,
             class Controller *ctrl_ptr, const uint32_t nvme_qp_num,
             const uint64_t dma_ioaddr, const uint64_t lba_start, const uint32_t num_lba_blocks) {
    // uint32_t max_transfer_lba_blocks = ctrl_ptr->info.max_data_size / 512;
    constexpr uint32_t max_transfer_lba_blocks = 1024 * 1024 * 2 / 512;

    uint8_t opcode;
    if constexpr (is_write) {
        opcode = NVM_IO_WRITE;
    } else {
        opcode = NVM_IO_READ;
    }

    uint32_t n_proc_blocks = 0;
    while (n_proc_blocks < num_lba_blocks) {
        if (n_proc_blocks + max_transfer_lba_blocks < num_lba_blocks) {
            nvm_cmd_process(gp, prps, &(ctrl_ptr->d_qps[nvme_qp_num]), opcode,
                            dma_ioaddr + (n_proc_blocks * 512), lba_start + n_proc_blocks,
                            (uint16_t)max_transfer_lba_blocks);
            n_proc_blocks += max_transfer_lba_blocks;
        } else {
            // Last issue
            nvm_cmd_process(gp, prps, &(ctrl_ptr->d_qps[nvme_qp_num]), opcode,
                            dma_ioaddr + (n_proc_blocks * 512), lba_start + n_proc_blocks,
                            (uint16_t)(num_lba_blocks - n_proc_blocks));
            break;
        }
    }
}

#define bam_find_arr_idx()                                                                                      \
    struct bam_array_d *bam_struct = &(bam_struct_in[nvme_dev_id]);                                             \
    uint32_t arr_select;                                                                                        \
    for (arr_select = 0; arr_select < bam_struct->num_bam_array; ++arr_select) {                                \
        if ((bam_struct->bam_array_nvme_start_byte[arr_select] <= byte_start) &&                                \
            (bam_struct->bam_array_nvme_end_byte[arr_select] + 1 >= byte_start + num_bytes)) {                  \
            break;                                                                                              \
        }                                                                                                       \
    }                                                                                                           \
    force_assert(arr_select < bam_struct->num_bam_array);                                                       \
    range_d_t<bam_t> *bam_range = bam_struct->bam_range[arr_select];                                            \
    /*array_d_t<bam_t> *bam_array = bam_struct->bam_array[arr_select];*/                                        \
    uint64_t bptr_idx_start = (byte_start - bam_struct->bam_array_nvme_start_byte[arr_select]) / sizeof(bam_t); \
    uint64_t bptr_idx_end_next = bptr_idx_start + (num_bytes / sizeof(bam_t));                                  \
    constexpr uint32_t bam_page_size = 4096;                                                                    \
    constexpr uint32_t num_elem_per_page = bam_page_size / sizeof(bam_t);                                       \
    uint32_t shift = __popc(num_elem_per_page - 1);

template <bool is_write>
__forceinline__ __device__ void
cache_copy_data(class gio *gp, const uint32_t nvme_dev_id, const uint32_t nvme_qp_id, struct bam_array_d *bam_struct_in,
                const void *dma_buf_vaddr, const uint64_t byte_start, const uint64_t num_bytes) {
    force_assert(byte_start % sizeof(bam_t) == 0);
    force_assert(num_bytes % sizeof(bam_t) == 0);

    // Use BaM cache
    bam_find_arr_idx();

    // printf("%s: arr_select: %u, bam_array: %p, bam_range: %p, byte_start: %lu, num_bytes: %lu, idx_start: %lu, idx_end_next: %lu\n",
    //        __func__, arr_select, bam_array, bam_range, byte_start, num_bytes, bptr_idx_start, bptr_idx_end_next);

    // Read the page only once
    uint32_t cross_page_boundary_count = (uint32_t)((bptr_idx_end_next - 1) >> shift) - (uint32_t)(bptr_idx_start >> shift);
    uint64_t bytes = 0;
    uint64_t copy_bytes = 0;
    char *dest = nullptr;
    char *src = nullptr;
    uint64_t lock_idx = bptr_idx_start;
    uint64_t page_id;

    for (uint32_t i = 0; i <= cross_page_boundary_count; ++i) {
        page_id = bam_range->get_page(lock_idx);

        uint64_t base;
        // page, count, dirty, ctrl, queue
        if constexpr (is_write) {
            base = bam_range->acquire_page(page_id, 1, true, nvme_dev_id, nvme_qp_id);
        } else {
            base = bam_range->acquire_page(page_id, 1, false, nvme_dev_id, nvme_qp_id);
        }

        void *temp = (void *)bam_range->get_cache_page_addr(base);

        if (cross_page_boundary_count == 0) {
            // only page
            uint32_t bytes_from_boundary = (bptr_idx_start & (num_elem_per_page - 1)) * sizeof(bam_t);
            dest = (char *)dma_buf_vaddr;
            src = (char *)temp + bytes_from_boundary;
            copy_bytes = num_bytes;
        } else if (i == 0) {
            // first page, copy until page boundary
            uint32_t bytes_from_boundary = (bptr_idx_start & (num_elem_per_page - 1)) * sizeof(bam_t);
            dest = (char *)dma_buf_vaddr;
            src = (char *)temp + bytes_from_boundary;
            copy_bytes = bam_page_size - bytes_from_boundary;
            bytes += copy_bytes;
        } else if (i == cross_page_boundary_count) {
            // last page
            dest = (char *)dma_buf_vaddr + bytes;
            src = (char *)temp;
            copy_bytes = num_bytes - bytes;
        } else {
            // full page copy in between
            dest = (char *)dma_buf_vaddr + bytes;
            src = (char *)temp;
            copy_bytes = bam_page_size;
            bytes += copy_bytes;
        }

        if constexpr (is_write) {
            my_memcpy(src, dest, copy_bytes);
        } else {
            // memcpy(dest, src, copy_bytes);
            my_memcpy(dest, src, copy_bytes);
        }

        // bam_array->release_raw(lock_idx);
        bam_range->release_page(page_id, 1);
        // lock_idx += copy_bytes / sizeof(bam_t);
        lock_idx += num_elem_per_page;
    }
}

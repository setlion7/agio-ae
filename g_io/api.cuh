#pragma once

#include "ds.cuh"
#include "macro.h"
#include "memalloc.cuh"
#include "submit.cuh"

// Fill iocb without any modifications
__forceinline__ __device__ void
_fill_iocb(class gio *gp, class iocb *cb,
           const uint32_t opcode_id, const uint64_t cid, const uint32_t devid,
           void *addr, const uint64_t start_byte, const uint64_t n_bytes, const gpua_tag_t tag) {
    gpua_opcode_t opcode;
    opcode.device_id = devid;
    opcode.opcode_id = opcode_id;

    cb->command_id = cid;
    cb->address = (uint64_t)addr;
    cb->start_byte = start_byte;
    cb->num_bytes = n_bytes;
    cb->opcode = opcode;
    cb->tag = tag;
}

// If *dma_buf_dptr is nullptr, allocate memory and return address
// Else, use *dma_buf_dptr
// Note the (void **)
__forceinline__ __device__ void
_build_iocb(class gio *gp, class iocb *cb,
            const uint32_t opcode_id, const uint64_t cid, const uint32_t devid,
            void **dma_buf_dptr_loc, const uint64_t start_byte, const uint64_t n_bytes, const gpua_tag_t tag) {

    void *address = (void *)((uint64_t)dma_buf_dptr_loc | 0x1); // possible delayed allocation
    _fill_iocb(gp, cb, opcode_id, cid, devid, address, start_byte, n_bytes, tag);
}

#define G_NOOP 0b01U
#define G_READ 0b00U
#define G_WRITE 0b10U
typedef uint32_t g_build_flag_t;

// If direct read, return the alignment required in bytes (NVMe LBA alignment) in cb->align_add
// Set cb->valid to true
__forceinline__ __device__ void
g_build_rtcb(class gio *gp, struct iocb *cb, const uint64_t cid, const gpua_tag_t tag, const uint32_t devid,
             void **dma_buf_vaddr, uint64_t start_byte, uint64_t n_bytes, const g_build_flag_t flags) {
    if (!cb->no_notify) {
        force_assert_printf(tag > 0, "%s: tag must be larger than zero to use the wait queue.\n", __func__);
    }
    uint32_t opcode_id;
    if (flags & G_NOOP) {
        opcode_id = GPU_OPCODE_NOOP;
        start_byte = 0;
        n_bytes = 0;
    } else if (flags & G_WRITE) {
        // write
        if (cb->direct_io) {
            // write direct
            force_assert(start_byte % G_LBA_SIZE == 0);
            force_assert(n_bytes % G_LBA_SIZE == 0);
            opcode_id = GPU_OPCODE_WRITE_DIRECT;
        } else {
            // write via cache
            opcode_id = GPU_OPCODE_WRITE_CACHE;
        }

    } else {
        // read
        if (cb->direct_io) {
            // read direct
            // force_assert(start_byte % G_LBA_SIZE == 0);
            // force_assert(n_bytes % G_LBA_SIZE == 0);

            // The buffer must be larger if not aligned, and return
            // the additional padding after the start address inside cb->align_add

            // Align start_byte and num_bytes to LBA boundary
            const uint64_t nvme_start_byte = start_byte & ~(G_CROSS_LBA_MASK); // Start must be LBA-aligned
            const uint32_t remainder_byte = start_byte & G_CROSS_LBA_MASK;
            const uint32_t nvme_bytes_to_read = ((start_byte + (uint64_t)n_bytes - 1) | G_CROSS_LBA_MASK) + 1 - nvme_start_byte;
            cb->align_add = remainder_byte;
            opcode_id = GPU_OPCODE_READ_DIRECT;
            start_byte = nvme_start_byte;
            n_bytes = nvme_bytes_to_read;
        } else {
            // read via cache
            cb->align_add = 0;
            opcode_id = GPU_OPCODE_READ_CACHE;
        }
    }
    _build_iocb(gp, cb, opcode_id, cid, devid, dma_buf_vaddr, start_byte, n_bytes, tag);
    cb->valid = true;
}

// Submit the iocb and process asynchronously (via SQ)
// Return values: 0: issued successfully, 1: no-op, -1: retry later
__forceinline__ __device__ int32_t
g_submit_async_n(class gio *gp, struct iocb *cb, uint32_t n_ios) {
    for (uint32_t i = 0; i < n_ios; ++i) {
        if (cb[i].valid == false || (cb[i].submitted == true))
            return 1;
    }
    int32_t ret;
    if (cb->fixed_queue) {
        USER_GID;
        const uint32_t qid = gid % gp->n_sq;
        ret = _gpu_submit_async(gp, qid, cb, n_ios);
    } else {
        const uint32_t qid = clock64() % gp->n_sq;
        ret = _gpu_submit_async(gp, qid, cb, n_ios);
    }
    return ret;
}

// Submit the iocb and process synchronously (caller processes the cb)
// Return values: 0: issued successfully, 1: no-op, -1: retry later
__forceinline__ __device__ int32_t
g_submit(class gio *gp, struct iocb *cb) {
    int32_t ret;
    if (cb->fixed_queue) {
        USER_GID;
        const uint32_t qid = gid % gp->n_sq;
        ret = _gpu_submit(gp, qid, cb, 1);
    } else {
        const uint32_t qid = clock64() % gp->n_sq;
        ret = _gpu_submit(gp, qid, cb, 1);
    }
    return ret;
}

#define G_WAIT_FIXED 0b100U
// check fixed-queue? random-queue | block-until-any? check-once | 0
typedef uint32_t g_wait_flag_t;

// Return a completed cid on success
// If a cid is being returned, the caller must process it as the cid will be removed from the wait queue
// Return negative on error
__forceinline__ __device__ int64_t
g_wait_any(class gio *gp, const g_wait_flag_t flags) {
    int64_t ret_cid = 0;
    // attempt to block until any I/O is available
    if (flags & G_WAIT_FIXED) {
        // wait in fixed queue (determined by gid)
        USER_GID;
        const uint32_t waitq_num = gid % gp->wait_ds.num_lib_waitq;
        ret_cid = gp->wait_ds._gpu_wait_any_tag(waitq_num, 0);
    } else {
        // wait in random queue
        const uint32_t waitq_num = clock64() % gp->wait_ds.num_lib_waitq;
        ret_cid = gp->wait_ds._gpu_wait_any_tag(waitq_num, 0);
    }
    return ret_cid;
}

__forceinline__ __device__ int64_t
g_wait_tag(class gio *gp, const uint32_t tag, const g_wait_flag_t flags) {
    int64_t ret_cid = 0;
    // attempt to block until any I/O is available
    if (flags & G_WAIT_FIXED) {
        // wait in fixed queue (determined by gid)
        USER_GID;
        const uint32_t waitq_num = gid % gp->wait_ds.num_lib_waitq;
        // constexpr uint32_t lookout = 256;
        // ret_cid = gp->wait_ds._gpu_wait_any_limited(gp, waitq_num, lookout);
        ret_cid = gp->wait_ds._gpu_wait_any_tag(waitq_num, tag);

    } else {
        // wait in random queue
        const uint32_t waitq_num = clock64() % gp->wait_ds.num_lib_waitq;
        ret_cid = gp->wait_ds._gpu_wait_any_tag(waitq_num, tag);
    }
    return ret_cid;
}

// Check for status of specific cid
// Return cid if cid is complete, else -1
__forceinline__ __device__ int64_t
g_check_cid(class gio *gp, int64_t cid) {
    // attempt to block until cid is complete
    uint32_t stage_id = (simple_hash_len4((uint8_t *)&cid)) % gp->n_stage;
    class lstage *sp = &(gp->stage[stage_id]);
    int32_t ret = sp->_check_block_64(cid);
    if (ret)
        cid = -1;
    return cid;
}

// Check for remaining I/O in SQ and WQ
// Return 0 if no remaining I/O
__forceinline__ __device__ int32_t
g_check_pending(class gio *gp, int32_t idx = -1) {
    int32_t ret_status;
    uint32_t qid;
    if (idx < 0) {
        USER_TID;
        qid = tid % gp->n_sq;
    } else {
        qid = idx % gp->n_sq;
    }
    uint32_t issue_pending = gp->s_queue[qid].lq.get_n_pending();
    uint32_t wait_pending = gp->wait_ds.wait_queue[qid].lq.get_n_pending();
    if ((issue_pending == 0) && (wait_pending == 0)) {
        ret_status = 0;
    } else {
        ret_status = issue_pending + wait_pending;
    }
    return ret_status;
}

class g_api {
    class gio *gp;

    public:
    __device__ g_api(decltype(gp) gp_in) : gp(gp_in) {
    }
    __device__ ~g_api() {
    }

    __forceinline__ __device__ void
    get_mem(memalloc_dptr_t *ret_dptr, const uint64_t req_size) {
        gp->mptr->get_mem(ret_dptr, req_size);
    }
    __forceinline__ __device__ void
    free_mem(const void *dptr) {
        gp->mptr->free_mem(dptr);
    }

    typedef uint32_t g_api_flags_t;
#define G_API_NOOP (1 << 0)
#define G_API_WRITE (1 << 3)
#define G_API_NOTIFY (1 << 4)
#define G_API_NO_NOTIFY (1 << 5)
#define G_API_DIRECT (1 << 6)
#define G_API_FIXED_QUEUE (1 << 7)

    // Build control block.
    // Default options: use notification (wait queue) with tag=1.
    // To ignore the IO progress after issue (e.g., write or correlated read), set G_API_NO_NOTIFY.
    // To perform direct IO (i.e., no cache), set G_API_DIRECT.
    __forceinline__ __device__ void
    build_rtcb(struct iocb *cb, const uint64_t cid, const uint32_t devid,
               void **dma_buf_vaddr, const uint64_t start_byte, const uint64_t n_bytes,
               g_api_flags_t flags = 0) {
        build_rtcb_with_tag(cb, cid, 1, devid, dma_buf_vaddr, start_byte, n_bytes, flags);
    }

    __forceinline__ __device__ void
    build_rtcb_with_tag(struct iocb *cb, const uint64_t cid, gpua_tag_t tag, const uint32_t devid,
                        void **dma_buf_vaddr, const uint64_t start_byte, const uint64_t n_bytes,
                        g_api_flags_t flags = 0) {
        force_assert(tag > 0);
        if (flags & G_API_NO_NOTIFY) {
            tag = 0;
            cb->no_notify = true;
        } else {
            cb->no_notify = false;
        }
        if (flags & G_API_DIRECT) {
            cb->direct_io = true;
        } else {
            cb->direct_io = false;
        }
        g_build_flag_t buildflag = 0;
        if (flags & G_API_NOOP) {
            buildflag |= G_NOOP;
        } else if (flags & G_API_WRITE) {
            buildflag |= G_WRITE;
        }
        g_build_rtcb(gp, cb, cid, tag, devid, dma_buf_vaddr, start_byte, n_bytes, flags);
    }

    // Issue IO according to opcode (read/write) of cb
    __forceinline__ __device__ int32_t
    submit_async(class iocb *cb, const g_api_flags_t flags = 0) {
        if (flags & G_API_FIXED_QUEUE)
            cb->fixed_queue = true;
        else
            cb->fixed_queue = false;
        return g_submit_async_n(gp, cb, 1);
    }
    __forceinline__ __device__ int32_t
    submit_async_n(struct iocb *cb, uint32_t n_ios, const g_api_flags_t flags = 0) {
        if (flags & G_API_FIXED_QUEUE)
            cb->fixed_queue = true;
        else
            cb->fixed_queue = false;
        return g_submit_async_n(gp, cb, n_ios);
    }
    __forceinline__ __device__ int32_t
    submit(struct iocb *cb, const g_api_flags_t flags = 0) {
        if (flags & G_API_FIXED_QUEUE)
            cb->fixed_queue = true;
        else
            cb->fixed_queue = false;
        return g_submit(gp, cb);
    }

    // Initiate async IO read
    __forceinline__ __device__ int32_t
    aio_read_n(class iocb *cb, const uint32_t n_cb, const g_api_flags_t flags = 0) {
        for (uint32_t i = 0; i < n_cb; ++i) {
            if (cb[i].direct_io)
                cb[i].opcode.opcode_id = GPU_OPCODE_READ_DIRECT;
            else
                cb[i].opcode.opcode_id = GPU_OPCODE_READ_CACHE;
        }
        if (flags & G_API_FIXED_QUEUE)
            cb->fixed_queue = true;
        else
            cb->fixed_queue = false;
        return g_submit_async_n(gp, cb, n_cb);
    }
    // Initiate async IO write
    __forceinline__ __device__ int32_t
    aio_write_n(class iocb *cb, const uint32_t n_cb, const g_api_flags_t flags = 0) {
        for (uint32_t i = 0; i < n_cb; ++i) {
            if (cb[i].direct_io)
                cb[i].opcode.opcode_id = GPU_OPCODE_WRITE_DIRECT;
            else
                cb[i].opcode.opcode_id = GPU_OPCODE_WRITE_CACHE;
        }
        if (flags & G_API_FIXED_QUEUE)
            cb->fixed_queue = true;
        else
            cb->fixed_queue = false;
        return g_submit_async_n(gp, cb, n_cb);
    }
    __forceinline__ __device__ int32_t
    aio_read(class iocb *cb, const g_api_flags_t flags = 0) {
        return aio_read_n(cb, 1, flags);
    }
    __forceinline__ __device__ int32_t
    aio_write(class iocb *cb, const g_api_flags_t flags = 0) {
        return aio_write_n(cb, 1, flags);
    }

    // Wait for a specific IO
    __forceinline__ __device__ int64_t
    wait_cid(const uint64_t cid, const g_api_flags_t flags = 0) {
        force_assert(!(flags & G_API_FIXED_QUEUE));
        return g_check_cid(gp, cid);
    }
    // Wait for any IO with matching tag
    __forceinline__ __device__ int64_t
    wait_tag(const uint32_t tag, const g_api_flags_t flags = 0) {
        g_wait_flag_t newflags = 0;
        if (flags & G_API_FIXED_QUEUE)
            newflags |= G_WAIT_FIXED;
        else
            newflags |= G_WAIT_FIXED;
        return g_wait_tag(gp, tag, newflags);
    }
    // Wait for a specific cid
    __forceinline__ __device__ int64_t
    wait_any(const g_api_flags_t flags = 0) {
        g_wait_flag_t newflags = 0;
        if (flags & G_API_FIXED_QUEUE)
            newflags |= G_WAIT_FIXED;
        else
            newflags |= G_WAIT_FIXED;
        return g_wait_any(gp, newflags);
    }

    // Check pending items in the queue
    __forceinline__ __device__ int32_t
    check_pending(int32_t qid = -1) {
        return g_check_pending(gp, qid);
    }
};

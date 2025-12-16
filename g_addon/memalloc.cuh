#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <limits>
#include <vector>
#include <thread>
#include <chrono>

#include "helper_headers/json.hpp"

#include "helper_headers/fassert.cuh"
#include "helper_headers/gpuErrChk.h"
#include "helper_headers/gpuHash.cuh"

#include "buffer.cuh"

#define MEMALLOC_USED 0xABAFA3A1U
#define MEMALLOC_FREE 0xCDED2D5DU
#define MEMALLOC_MAX_QUEUES 64
#define QUEUE_TABLE_LEN_MAX 32

constexpr bool MEMALLOC_QUEUE_GROUPBY_SIZE = true;

typedef uint64_t memalloc_ioaddr_t;
typedef void *memalloc_dptr_t;
typedef uint32_t memalloc_status_t;

template <typename T>
struct __align__(4) mem_padded_atomic {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[4];
    };
};

struct __align__(32) ioaddr_lookup {
    memalloc_dptr_t dptr_start;
    memalloc_dptr_t dptr_end;
    memalloc_ioaddr_t ioaddr_start;
    uint64_t size;
    uint32_t qid;
};

struct __align__(128) mem_queue {
    memalloc_dptr_t __align__(32) dma_buffer_dptr;
    memalloc_ioaddr_t dma_buffer_ioaddr;
    uint64_t dma_buffer_num_pages;
    uint64_t dma_buffer_size;
    uint32_t dma_buffer_page_size;
    cuda::atomic<uint32_t, cuda::thread_scope_device> not_empty_cnt;
    struct mem_padded_atomic<uint32_t> mem_next;

    cuda::atomic<memalloc_status_t, cuda::thread_scope_device> *entry_status;
    uint32_t *slot_num_cont;
    uint32_t my_idx;
    bool size_leader; // true if this queue is the leader within the same page_size queues

    __device__ void mem_queue_init_d(uint32_t idx, bool leader) {
        not_empty_cnt.store(0, cuda::memory_order_relaxed);
        mem_next.value.store(0, cuda::memory_order_relaxed);
        force_assert(dma_buffer_num_pages != 0);
        force_assert(dma_buffer_page_size != 0);
        dma_buffer_size = dma_buffer_num_pages * dma_buffer_page_size;
        my_idx = idx;
        size_leader = leader;
    }
    __device__ void mem_queue_fini_d() {
        printf("%s: mq[%u], mem slot_size: %u, not_empty_cnt: %u, mem_next: %u\t",
               __func__, my_idx, dma_buffer_page_size, not_empty_cnt.load(), mem_next.value.load());
        if (my_idx % 2 == 1)
            printf("\n");
    }

    __host__ void mem_queue_init(uint64_t dma_n_pages, uint32_t dma_page_size,
                                 memalloc_dptr_t dma_dptr, memalloc_ioaddr_t dma_ioaddr) {
        static_assert(std::is_standard_layout_v<mem_queue> == true);
        static_assert(std::is_trivially_copyable_v<mem_queue> == true);

        cudaErrChk(cudaMemset(dma_dptr, 0, dma_n_pages * dma_page_size));
        std::remove_pointer_t<decltype(this)> mq;
        mq.dma_buffer_num_pages = dma_n_pages;
        mq.dma_buffer_page_size = dma_page_size;
        mq.dma_buffer_dptr = dma_dptr;
        mq.dma_buffer_ioaddr = dma_ioaddr;

        std::vector<cuda::atomic<uint32_t, cuda::thread_scope_device>> status_h(dma_n_pages);
        for (size_t i = 0; i < status_h.size(); ++i) {
            status_h[i] = MEMALLOC_FREE;
        }

        cudaErrChk(cudaMalloc(&(mq.entry_status), sizeof(status_h[0]) * dma_n_pages));
        cudaErrChk(cudaMemcpy(mq.entry_status, status_h.data(), sizeof(status_h[0]) * dma_n_pages, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMalloc(&(mq.slot_num_cont), sizeof(uint32_t) * dma_n_pages));
        cudaErrChk(cudaMemset(mq.slot_num_cont, 0, sizeof(uint32_t) * dma_n_pages));

        memcpy(this, &mq, sizeof(*this));
    }
    __host__ void mem_queue_fini() {
        std::remove_pointer_t<decltype(this)> mq;
        memcpy(reinterpret_cast<char *>(&mq), this, sizeof(*this));

        std::vector<memalloc_status_t> status_h(mq.dma_buffer_num_pages);
        cudaErrChk(cudaMemcpy(status_h.data(), mq.entry_status, sizeof(status_h[0]) * mq.dma_buffer_num_pages, cudaMemcpyDeviceToHost));
        size_t n_not_free = 0;
        for (uint64_t slot = 0; slot < mq.dma_buffer_num_pages; ++slot) {
            if (status_h[slot] != (memalloc_status_t)MEMALLOC_FREE) {
                if (n_not_free < 4)
                    fprintf(stderr, "%s: memalloc not free! mqid: %u (page size: %u), status: %X, slot: %lu (total slots: %lu)\n",
                            __func__, mq.my_idx, mq.dma_buffer_page_size, status_h[slot], slot, mq.dma_buffer_num_pages);
                ++n_not_free;
            }
        }
        if (n_not_free > 0) {
            fprintf(stderr, "%s: memalloc not free! mqid: %u (page size: %u), total n_slots not free: %lu\n",
                    __func__, mq.my_idx, mq.dma_buffer_page_size, n_not_free);
            exit(EXIT_FAILURE);
        }
        cudaErrChk(cudaFree(mq.entry_status));
        cudaErrChk(cudaFree(mq.slot_num_cont));
    }
};

typedef struct ioaddr_lookup memalloc_ioaddr_lookup_t;
typedef uint32_t memalloc_queue_lookup_t;
class __align__(32) memalloc {
    public:
    struct mem_queue *m_queue_d;
    cuda::atomic<uint32_t, cuda::thread_scope_device> mem_pressure;
    uint32_t num_mem_queue;
    uint32_t num_mem_parallel;
    uint64_t num_pages[MEMALLOC_MAX_QUEUES];
    uint32_t page_size[MEMALLOC_MAX_QUEUES];
    memalloc_ioaddr_lookup_t ioaddr_table[MEMALLOC_MAX_QUEUES];
    memalloc_queue_lookup_t queue_table[QUEUE_TABLE_LEN_MAX];

    __forceinline__ __device__ void
    memalloc_init_d() {
        mem_pressure.store(1);
        if constexpr (MEMALLOC_QUEUE_GROUPBY_SIZE == false) {
            for (uint32_t i = 0; i < num_mem_parallel; ++i) {
                bool leader = false;
                if (i == 0) {
                    leader = true;
                }
                for (uint32_t j = 0; j < num_mem_queue; ++j) {
                    uint32_t qid = (i * num_mem_queue) + j;
                    new (&(m_queue_d[qid])) struct mem_queue;
                    m_queue_d[qid].mem_queue_init_d(qid, leader);
                }
            }
        } else {
            for (uint32_t i = 0; i < num_mem_queue; ++i) {
                for (uint32_t j = 0; j < num_mem_parallel; ++j) {
                    bool leader = false;
                    if (j == 0) {
                        leader = true;
                    }
                    uint32_t qid = (i * num_mem_parallel) + j;
                    new (&(m_queue_d[qid])) struct mem_queue;
                    m_queue_d[qid].mem_queue_init_d(qid, leader);
                }
            }
        }
    }
    __forceinline__ __device__ void
    memalloc_fini_d() {
        for (uint32_t i = 0; i < num_mem_queue * num_mem_parallel; ++i) {
            m_queue_d[i].mem_queue_fini_d();
            m_queue_d[i].~mem_queue();
        }
    }

    __host__ void
    memalloc_init_h(const nvm_ctrl_t *sel_ctrl, const nlohmann::json in_js,
                    std::vector<std::shared_ptr<nvm_dma_t>> &dma_shared_ptr,
                    std::vector<std::tuple<uint32_t, void *>> &allocator) {
        static_assert(std::is_standard_layout_v<memalloc> == true);
        static_assert(std::is_trivially_copyable_v<memalloc> == true);

        std::vector<uint64_t> queue_len_vec;
        std::vector<uint32_t> page_size_vec;
        uint32_t alloc_method = 0;

        try {
            num_mem_queue = in_js.at("num_mem_queues");
            num_mem_parallel = in_js.at("num_mem_parallel");
            queue_len_vec = in_js.at("queue_len").get<std::vector<uint64_t>>();
            page_size_vec = in_js.at("slot_size").get<std::vector<uint32_t>>();
            alloc_method = in_js.at("alloc_method");
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            force_assert(false);
        }
        force_assert(num_mem_queue * num_mem_parallel <= MEMALLOC_MAX_QUEUES);
        force_assert(num_mem_queue == queue_len_vec.size());
        force_assert(queue_len_vec.size() == page_size_vec.size());

        for (uint32_t i = 0; i < 32; ++i) {
            queue_table[i] = std::numeric_limits<memalloc_queue_lookup_t>::max();
        }

        if constexpr (MEMALLOC_QUEUE_GROUPBY_SIZE == false) {
            for (uint32_t i = 0; i < num_mem_parallel; ++i) {
                for (uint32_t j = 0; j < num_mem_queue; ++j) {
                    uint32_t idx = i * num_mem_queue + j;
                    num_pages[idx] = queue_len_vec[j];
                    page_size[idx] = page_size_vec[j];
                    // force_assert(std::has_single_bit(page_size[idx]) == true);
                    force_assert(g_IsPowerOfTwo(page_size[idx]) == true);
                    queue_table[(uint32_t)log2(page_size[idx])] = j;
                    // printf("%s: num_pages: %lu, page_size: %u, log: %u, j: %u\n",
                    //        __func__, num_pages[idx], page_size[idx], (uint32_t)log2(page_size[idx]), j);
                }
            }
        } else { // GROUPBY_SIZE == true
            for (uint32_t i = 0; i < num_mem_queue; ++i) {
                for (uint32_t j = 0; j < num_mem_parallel; ++j) {
                    uint32_t idx = i * num_mem_parallel + j;
                    num_pages[idx] = queue_len_vec[i];
                    page_size[idx] = page_size_vec[i];
                    force_assert(g_IsPowerOfTwo(page_size[idx]) == true);
                    queue_table[(uint32_t)log2(page_size[idx])] = i * num_mem_parallel;
                    // printf("%s: idx: %u, num_pages: %lu, page_size: %u, log: %u, j: %u\n",
                    //        __func__, idx, num_pages[idx], page_size[idx], (uint32_t)log2(page_size[idx]), j);
                }
            }
        }

        // for (uint32_t i = 0; i < QUEUE_TABLE_LEN_MAX; ++i) {
        //     printf("%s: queue_table[%u]: %u\n", __func__, i, queue_table[i]);
        // }

        // Fill in the queue_table gaps (use a slot larger)
        uint32_t qt_idx = 0;
        for (uint32_t i = 0; i < QUEUE_TABLE_LEN_MAX; ++i) {
            if constexpr (MEMALLOC_QUEUE_GROUPBY_SIZE == false) {
                if (queue_table[i] == std::numeric_limits<memalloc_queue_lookup_t>::max()) {
                    queue_table[i] = qt_idx;
                } else if (qt_idx < num_mem_queue - 1) {
                    qt_idx++;
                }
            } else { // GROUPBY_SIZE
                if (queue_table[i] == std::numeric_limits<memalloc_queue_lookup_t>::max()) {
                    queue_table[i] = qt_idx;
                } else if (qt_idx < (num_mem_queue - 1) * num_mem_parallel) {
                    qt_idx += num_mem_parallel;
                }
            }
            // printf("%s: fill queue_table[%u]: %u, qt_idx: %u\n", __func__, i, queue_table[i], qt_idx);
        }

        std::vector<memalloc_ioaddr_lookup_t> io_lookup_vec(MEMALLOC_MAX_QUEUES);
        uint64_t total_dma_buffer_size = 0;
        std::vector<struct mem_queue> mq_temp_h(num_mem_queue * num_mem_parallel);
        uint32_t i;
        for (i = 0; i < num_mem_queue * num_mem_parallel; ++i) {
            uint64_t dma_buffer_size = (uint64_t)page_size[i] * (uint64_t)num_pages[i];
            total_dma_buffer_size += dma_buffer_size;

            memalloc_dptr_t dma_buffer_dptr;
            memalloc_ioaddr_t dma_buffer_ioaddr;
            bool alloc_dma = (alloc_method == 1) || (alloc_method == 2 && page_size[i] < 100000);
            if (alloc_dma) {
                uint32_t tempcnt = 0;
                while (true) {
                    dma_shared_ptr.push_back(createDma_cuda(sel_ctrl, dma_buffer_size));
                    if (dma_shared_ptr[i].get()->contiguous != true) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        dma_shared_ptr.pop_back();
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    } else {
                        break;
                    }
                    ++tempcnt;
                    force_assert_printf(tempcnt < 8,
                                        "%s: dma memory is not contiguous, request size: %lu\n",
                                        __func__, dma_buffer_size);
                }
                force_assert(dma_shared_ptr[i].get()->page_size == 4096);
                dma_buffer_dptr = dma_shared_ptr[i].get()->vaddr;
                force_assert((uint64_t)dma_buffer_dptr >> 63 == 0);
                dma_buffer_ioaddr = dma_shared_ptr[i].get()->ioaddrs[0];
                allocator.push_back(std::make_tuple(1, dma_buffer_dptr));
            } else {
                void *cmptr = nullptr;
                cudaErrChk(cudaMallocManaged(&cmptr, dma_buffer_size));
                force_assert(cmptr != nullptr);
                cudaMemLocation loc;
                loc.id = 0;
                loc.type = cudaMemLocationTypeDevice;
                cudaErrChk(cudaMemAdvise(cmptr, dma_buffer_size, cudaMemAdviseSetPreferredLocation, loc));
                cudaErrChk(cudaMemAdvise(cmptr, dma_buffer_size, cudaMemAdviseSetAccessedBy, loc));
                dma_buffer_dptr = cmptr;
                dma_buffer_ioaddr = 0;
                allocator.push_back(std::make_tuple(2, dma_buffer_dptr));
            }

            force_assert(page_size[i] % 8 == 0);
            mq_temp_h[i].mem_queue_init(num_pages[i], page_size[i], dma_buffer_dptr, dma_buffer_ioaddr);

            memalloc_ioaddr_lookup_t temp;
            temp.size = dma_buffer_size;
            temp.dptr_start = dma_buffer_dptr;
            temp.dptr_end = (char *)dma_buffer_dptr + temp.size;
            temp.ioaddr_start = dma_buffer_ioaddr;
            temp.qid = i;
            io_lookup_vec[i] = temp;
        }

        cudaErrChk(cudaMalloc(&m_queue_d, sizeof(struct mem_queue) * num_mem_queue * num_mem_parallel));
        cudaErrChk(cudaMemcpy(m_queue_d, mq_temp_h.data(), sizeof(mq_temp_h[0]) * num_mem_queue * num_mem_parallel, cudaMemcpyHostToDevice));

        for (; i < MEMALLOC_MAX_QUEUES; ++i) {
            memalloc_ioaddr_lookup_t temp;
            temp.size = 0;
            temp.dptr_start = (void *)std::numeric_limits<uint64_t>::max();
            temp.dptr_end = (void *)std::numeric_limits<uint64_t>::max();
            temp.ioaddr_start = std::numeric_limits<memalloc_ioaddr_t>::max();
            temp.qid = std::numeric_limits<uint32_t>::max();
            io_lookup_vec[i] = temp;
        }
        std::sort(io_lookup_vec.begin(), io_lookup_vec.end(),
                  [](const memalloc_ioaddr_lookup_t &a, const memalloc_ioaddr_lookup_t &b) {
                      return a.dptr_start < b.dptr_start;
                  });

        printf("%s: memalloc total_dma_buffer_size: %lu (%.3lf MiB)\n",
               __func__, total_dma_buffer_size, (double)total_dma_buffer_size / 1024.0 / 1024.0);

        memset(ioaddr_table, 0, sizeof(memalloc_ioaddr_lookup_t) * MEMALLOC_MAX_QUEUES);
        memcpy(ioaddr_table, io_lookup_vec.data(), sizeof(memalloc_ioaddr_lookup_t) * io_lookup_vec.size());
        // for (size_t i = 0; i < io_lookup_vec.size(); ++i) {
        //     printf("%s: memalloc ioaddr lookup: %p -> %lX, size: %lu\n",
        //            __func__, ioaddr_table[i].dptr_start, ioaddr_table[i].ioaddr_start, ioaddr_table[i].size);
        // }
    }
    __host__ void
    memalloc_fini_h(std::vector<std::shared_ptr<nvm_dma_t>> &dma_shared_ptr,
                    std::vector<std::tuple<uint32_t, void *>> &allocator) {
        std::vector<struct mem_queue> mq_temp_h(num_mem_queue * num_mem_parallel);
        cudaErrChk(cudaMemcpy(mq_temp_h.data(), m_queue_d, sizeof(mq_temp_h[0]) * num_mem_queue * num_mem_parallel, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < num_mem_queue * num_mem_parallel; ++i) {
            mq_temp_h[i].mem_queue_fini();
        }
        cudaErrChk(cudaFree(m_queue_d));

        for (size_t i = 0; i < dma_shared_ptr.size(); ++i) {
            dma_shared_ptr[i].reset();
        }
        for (size_t i = 0; i < allocator.size(); ++i) {
            if (std::get<0>(allocator[i]) == 2) {
                cudaErrChk(cudaFree(std::get<1>(allocator[i])));
            }
        }
    }

    __forceinline__ __device__ void
    _get_mem(memalloc_dptr_t *ret_dptr, const uint64_t req_size, const uint32_t memq_id);
    __forceinline__ __device__ void
    get_mem(memalloc_dptr_t *ret_dptr, const uint64_t req_size);
    __forceinline__ __device__ void
    free_mem(const void *dptr);

    __forceinline__ __device__ uint64_t
    get_ioaddr_from_dptr(const void *dptr);
    __forceinline__ __device__ uint32_t
    get_mqid_from_dptr(const void *dptr);

    __forceinline__ __host__ bool
    g_IsPowerOfTwo(uint32_t x) {
        return (x != 0) && ((x & (x - 1)) == 0);
    }
};

// Allocate memory. Set ret_dptr to nullptr if failed.
__forceinline__ __device__ void
memalloc::_get_mem(memalloc_dptr_t *ret_dptr, const uint64_t req_size, const uint32_t memq_id) {
    struct mem_queue *sel_queue = &(m_queue_d[memq_id]);
    const uint32_t page_size = sel_queue->dma_buffer_page_size;
    const uint64_t num_pages = sel_queue->dma_buffer_num_pages;

    uint32_t req_num_pages = req_size / page_size;
    if (req_size % page_size != 0)
        req_num_pages++;

    uint32_t redo_alloc_cnt = 0;
    uint32_t dma_slot;
redo_alloc:
    const uint32_t dma_slot1 = sel_queue->mem_next.value.fetch_add(req_num_pages, cuda::memory_order_relaxed) % num_pages;
    if (dma_slot1 + req_num_pages > num_pages) {
        dma_slot = sel_queue->mem_next.value.fetch_add(req_num_pages, cuda::memory_order_relaxed) % num_pages;
    } else {
        dma_slot = dma_slot1;
    }

    memalloc_dptr_t alloc_ptr = (char *)sel_queue->dma_buffer_dptr + ((uint64_t)dma_slot * (uint64_t)page_size);

    for (uint32_t i = 0; i < req_num_pages; ++i) {
        cuda::atomic<memalloc_status_t, cuda::thread_scope_device> *status = &(sel_queue->entry_status[dma_slot + i]);

        memalloc_status_t status_load = (*status).load(cuda::memory_order_acquire);
        if (status_load != (memalloc_status_t)MEMALLOC_FREE) {
            // printf("%s: unwinding... tidx: %u, mqid: %u, slot_size: %u, not free slot: %u, req size: %lu, req slots: %u, status: %X, mem_next: %u\n",
            //        __func__, threadIdx.x, memq_id, page_size, dma_slot, req_size, req_num_pages,
            //        status->load(cuda::memory_order_relaxed),
            //        sel_queue->mem_next.value.load(cuda::memory_order_relaxed));

            for (int32_t j = (int32_t)i - 1; j >= 0; --j) {
                cuda::atomic<memalloc_status_t, cuda::thread_scope_device> *status = &(sel_queue->entry_status[dma_slot + j]);
                (*status).store((memalloc_status_t)MEMALLOC_FREE, cuda::memory_order_relaxed);
            }

            redo_alloc_cnt++;
            if (redo_alloc_cnt > 8) {
                *ret_dptr = nullptr;
                const uint32_t not_empty_cnt = sel_queue->not_empty_cnt.fetch_add(1, cuda::memory_order_relaxed);
                if (not_empty_cnt == 64 && (sel_queue->size_leader == true)) {
                    uint32_t old = mem_pressure.fetch_add(1, cuda::memory_order_relaxed);
                    // printf("%s: qid: %u (page size: %u), not_empty_cnt: %u, req_size: %lu, req_num_pages: %u, old pressure: %u, initial increase mem pressure!\n",
                    //        __func__, sel_queue->my_idx, page_size, not_empty_cnt, req_size, req_num_pages, old);
                } else if ((not_empty_cnt > 0) && (not_empty_cnt % 16384 == 0) && (sel_queue->size_leader == true)) {
                    uint32_t old = mem_pressure.fetch_add(1, cuda::memory_order_relaxed);
                    // printf("%s: qid: %u (page size: %u), not_empty_cnt: %u, req_size: %lu, req_num_pages: %u, old pressure: %u, increase mem pressure!\n",
                    //        __func__, sel_queue->my_idx, page_size, not_empty_cnt, req_size, req_num_pages, old);
                }
                return;
            }

            __nanosleep(128 * redo_alloc_cnt); //
            goto redo_alloc;
        }

        if (i + 1 == req_num_pages) {
            *ret_dptr = alloc_ptr;
            sel_queue->slot_num_cont[dma_slot] = req_num_pages;
        }

        memalloc_status_t tstatus = status->exchange((memalloc_status_t)(MEMALLOC_USED ^ (uint32_t)((uint64_t)alloc_ptr)), cuda::memory_order_release);
        force_assert_printf(tstatus == MEMALLOC_FREE, "%s: exchange status not free, got %X\n", __func__, tstatus);
    }
}

__forceinline__ __device__ void
memalloc::get_mem(memalloc_dptr_t *ret_dptr, const uint64_t req_size) {
    force_assert(req_size != 0);
    // find qid w/o parallel
    uint32_t qid = 0;
    bool pow_two = false;
    if (__popcll(req_size) == 1) {
        pow_two = true;
    }
    int temp = __clzll(req_size);

    temp = 63 - temp;
    if (!pow_two)
        temp++;
    if (temp >= QUEUE_TABLE_LEN_MAX) {
        temp = QUEUE_TABLE_LEN_MAX - 1;
    }
    qid = queue_table[temp];

    // find qid w/ parallel
    // uint32_t mult = blockIdx.x % num_mem_parallel;
    const uint32_t mult = clock64() % num_mem_parallel;
    uint32_t mqid;
    if constexpr (MEMALLOC_QUEUE_GROUPBY_SIZE == false) {
        mqid = qid + (mult * num_mem_queue);
    } else {
        mqid = qid + mult;
    }
    // printf("%s: req_size: %lu, pow_two: %s, temp: %u, mult: %u, (original) qid: %u, mqid: %u\n",
    //        __func__, req_size, pow_two ? "true" : "false", temp, mult, qid, mqid);

    // force_assert(mqid < (num_mem_queue * num_mem_parallel));
    _get_mem(ret_dptr, req_size, mqid);
}

__forceinline__ __device__ void
memalloc::free_mem(const void *dptr) {
    const uint32_t memalloc_id = get_mqid_from_dptr(dptr);

    struct mem_queue *sel_queue = &(m_queue_d[memalloc_id]);
    const uint32_t page_size = sel_queue->dma_buffer_page_size;
    uint64_t diff = (uint64_t)((char *)dptr - (char *)sel_queue->dma_buffer_dptr);
    uint32_t slot = (uint64_t)diff / (uint64_t)page_size;
    force_assert((uint64_t)diff % (uint64_t)page_size == 0);

    if (sel_queue->entry_status[slot].load(cuda::memory_order_acquire) != (memalloc_status_t)(MEMALLOC_USED ^ (uint32_t)((uint64_t)dptr))) {
        force_assert_printf(false,
                            "%s: first slot free error! tidx: %u, bidx: %u, memq: %u, base: %p, dptr: %p, slot: %u, data: %X, slot (page) size: %u, n_slots to free: %u\n",
                            __func__, threadIdx.x, blockIdx.x, memalloc_id, sel_queue->dma_buffer_dptr, dptr,
                            slot, sel_queue->entry_status[slot].load(cuda::memory_order_relaxed),
                            sel_queue->dma_buffer_page_size,
                            sel_queue->slot_num_cont[slot]);
    }
    sel_queue->entry_status[slot].store((memalloc_status_t)MEMALLOC_FREE, cuda::memory_order_release);

    uint32_t num_slots = sel_queue->slot_num_cont[slot];

    for (uint32_t i = 1; i < num_slots; ++i) {
        if (sel_queue->entry_status[slot + i].load(cuda::memory_order_acquire) != (memalloc_status_t)(MEMALLOC_USED ^ (uint32_t)((uint64_t)dptr))) {
            force_assert_printf(false,
                                "%s: double free error! tidx: %u, bidx: %u, memq: %u, base: %p, dptr: %p, slot: %u, i: %u, idx: %u, data: %X, num_slots: %u, slot (page) size: %u\n",
                                __func__, threadIdx.x, blockIdx.x, memalloc_id, sel_queue->dma_buffer_dptr, dptr,
                                slot, i, slot + i, sel_queue->entry_status[slot + i].load(cuda::memory_order_relaxed),
                                num_slots, sel_queue->dma_buffer_page_size);
        }
        sel_queue->entry_status[slot + i].store((memalloc_status_t)MEMALLOC_FREE, cuda::memory_order_release);
    }
}

__forceinline__ __device__ uint64_t
memalloc::get_ioaddr_from_dptr(const void *dptr) {
    uint64_t ioaddr = 0;
    uint32_t qid = 0;
    for (; qid < MEMALLOC_MAX_QUEUES; ++qid) {
        if ((ioaddr_table[qid].dptr_start <= dptr) && ((char *)ioaddr_table[qid].dptr_start + ioaddr_table[qid].size > dptr)) {
            int64_t diff = (uint64_t)ioaddr_table[qid].dptr_start - ioaddr_table[qid].ioaddr_start;
            ioaddr = (uint64_t)dptr - diff;
            break;
        }
    }
    force_assert_printf(ioaddr > 0, "%s: ioaddr: %lX, dptr: %p\n", __func__, ioaddr, dptr);
    return ioaddr;
}

__forceinline__ __device__ uint32_t
memalloc::get_mqid_from_dptr(const void *dptr) {
    uint32_t qid;
    uint32_t left = 0;
    uint32_t right = MEMALLOC_MAX_QUEUES - 1;
    while (left <= right) {
        // uint32_t mid = (left + right) / 2;
        uint32_t mid = left + (right - left) / 3;
        if ((ioaddr_table[mid].dptr_start <= dptr) && (dptr < ioaddr_table[mid].dptr_end)) {
            qid = ioaddr_table[mid].qid;
            break;
        } else if (dptr < ioaddr_table[mid].dptr_start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return qid;
}

class memalloc_w {
    public:
    class memalloc memalloc_h;
    class memalloc *memalloc_dp;

    std::vector<std::shared_ptr<nvm_dma_t>> dma_shared_ptr;
    std::vector<std::tuple<uint32_t, void *>> allocator;

    void memalloc_w_init_h(const nvm_ctrl_t *sel_ctrl, const nlohmann::json in_js) {
        memalloc_h.memalloc_init_h(sel_ctrl, in_js, dma_shared_ptr, allocator);

        cudaErrChk(cudaMalloc(&memalloc_dp, sizeof(*memalloc_dp)));
        cudaErrChk(cudaMemcpy(memalloc_dp, &memalloc_h, sizeof(*memalloc_dp), cudaMemcpyHostToDevice));
    }
    void memalloc_w_fini_h() {
        cudaErrChk(cudaFree(memalloc_dp));
        memalloc_h.memalloc_fini_h(dma_shared_ptr, allocator);
    }
};

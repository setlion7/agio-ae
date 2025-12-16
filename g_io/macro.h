#pragma once

#include <cstdint>

// Queue status values
// <--tag 8 bits-->|<--vec i 4 bits-->|<--vec total 4 bits-->|<--status 16 bits-->
// vec applies to SQ
#define ENTRY_NOOP 0x00001566U
#define ENTRY_EMPTY 0x0000CBABU
#define ENTRY_VALID 0x000021E7U
#define ENTRY_FILLING 0x0000E3AAU
#define ENTRY_PROCESSING 0x000065F3U
#define ENTRY_PROCESSED 0x0000345AU

// Interface operations
#define GPU_OPCODE_NOOP 0x03
#define GPU_OPCODE_READ_DIRECT 0x05
#define GPU_OPCODE_READ_CACHE 0xAA
#define GPU_OPCODE_WRITE_DIRECT 0x06
#define GPU_OPCODE_WRITE_CACHE 0xBB

#define AT_OR_LARGE_MULT(X, M) ((X - 1) | (M - 1)) + 1
#define G_CROSS_LBA_MASK (512UL - 1)
#define G_LBA_SIZE 512U

// Wait
#define GPU_WAIT_ERROR -1L
#define GPU_WAIT_NOT_FOUND -5L
#define GPU_WAIT_TOO_MUCH -6L

// ID
#define LIB_TID const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
// #define USER_TID const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
#define USER_TID const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
#define LIB_GID const uint32_t gid = blockIdx.x;
// #define LIB_GID const uint32_t gid = tid / 32;
#define USER_GID const uint32_t gid = blockIdx.x;

// Queue length (must be larger then n_threads)
constexpr uint32_t n_sq_entries = (1 << 14);
constexpr uint32_t n_wq_entries = (1 << 15);
constexpr uint32_t n_stage_entries = (1 << 5);

// Macro for delayed memory allocation, if needed
// Bit 0 of addr_in must be 1 to attempt delayed memory allocation
// mptr, opcode, uint64_t (address to use in lower level), uint64_t (of void **), uint64_t
#define GPU_SUBMIT_MEMALLOC(mptr, opcode, addr_out, addr_in, n_bytes)      \
    bool try_delayed_alloc = (((uint64_t)addr_in & 0x01) == 0x01);         \
    if (try_delayed_alloc) {                                               \
        void **original = (void **)((uint64_t)addr_in & ~0x1UL);           \
        if (*original == nullptr) {                                        \
            force_assert(opcode != GPU_OPCODE_WRITE_CACHE);                \
            force_assert(opcode != GPU_OPCODE_WRITE_DIRECT);               \
            void *mem_dptr;                                                \
            if (opcode == GPU_OPCODE_READ_CACHE) {                         \
                mptr->get_mem(&mem_dptr, n_bytes);                         \
            } else if (opcode == GPU_OPCODE_READ_DIRECT) {                 \
                mptr->get_mem(&mem_dptr, n_bytes);                         \
            }                                                              \
            *original = mem_dptr; /*propagate to upper level*/             \
            addr_out = (uint64_t)mem_dptr;                                 \
        } else {                                                           \
            addr_out = (uint64_t)*original;                                \
        }                                                                  \
    } else {                                                               \
        addr_out = (uint64_t)*(void **)((uint64_t)addr_in & ~0x1UL);       \
    }                                                                      \
    if ((opcode == GPU_OPCODE_READ_DIRECT) && addr_out != 0) {             \
        addr_out = (uint64_t)mptr->get_ioaddr_from_dptr((void *)addr_out); \
    } else if (opcode == GPU_OPCODE_WRITE_DIRECT) {                        \
        addr_out = (uint64_t)mptr->get_ioaddr_from_dptr((void *)addr_out); \
    }

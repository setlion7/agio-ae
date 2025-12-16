#pragma once

template <typename T>
struct __align__(32) padded_atomic {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[64];
    };
};

template <typename T>
struct __align__(16) padded_atomic_status {
    union {
        cuda::atomic<T, cuda::thread_scope_device> value;
        uint8_t padding[16];
    };
};

struct gpua_opcode {
    uint16_t device_id;
    uint16_t opcode_id;
};
typedef struct gpua_opcode gpua_opcode_t;

typedef uint32_t gpua_tag_t;

class iocb {
    public:
    // struct io_s_entry start
    uint64_t command_id;
    uint64_t address;
    uint64_t start_byte;
    uint64_t num_bytes;
    gpua_opcode_t opcode;
    // struct io_s_entry done

    gpua_tag_t tag;
    uint32_t align_add;
    bool valid;
    bool submitted;

    bool direct_io;
    bool no_notify;
    bool fixed_queue;

    __device__ iocb()
        : address(0), tag(0), align_add(0),
          valid(false), submitted(false),
          direct_io(false), no_notify(false), fixed_queue(false) {
          };
    __device__ ~iocb(){};
};

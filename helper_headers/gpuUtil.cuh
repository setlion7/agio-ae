#include "fassert.cuh"

__forceinline__ __device__ void
fast_memcpy4(void *__restrict__ out, const void *__restrict__ in, const int bytes) {
    for (int i = 0; i < bytes / 4; ++i)
        ((int1 *)out)[i] = ((int1 *)in)[i];
}

__forceinline__ __device__ void
fast_memcpy8(void *__restrict__ out, const void *__restrict__ in, const int bytes) {
    if (((uintptr_t)in | (uintptr_t)out) % 8 == 0) {
        int chunks = bytes / 8;
        for (int i = 0; i < chunks; ++i)
            ((int2 *)out)[i] = ((int2 *)in)[i];
        if (bytes % 8)
            fast_memcpy4(&((int2 *)out)[chunks], &((int2 *)in)[chunks], bytes % 8);
    } else
        fast_memcpy4(out, in, bytes);
}

__forceinline__ __device__ void
fast_memcpy16(void *__restrict__ out, const void *__restrict__ in, const int bytes) {
    if (((uintptr_t)in | (uintptr_t)out) % 16 == 0) {
        int chunks = bytes / 16;
        for (int i = 0; i < chunks; ++i)
            ((int4 *)out)[i] = ((int4 *)in)[i];
        if (bytes % 16)
            fast_memcpy8(&((int4 *)out)[chunks], &((int4 *)in)[chunks], bytes % 16);
    } else
        fast_memcpy8(out, in, bytes);
}

__forceinline__ __device__ void
my_memcpy(void *__restrict__ dest, void *__restrict__ src, const int bytes) {
    // memcpy(dest, src, bytes);
    fast_memcpy16(dest, src, bytes);
    // fast_memcpy16_asm(dest, src, bytes);
}

__forceinline__ __device__ uint64_t
random_xorshift64(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

__device__ uint64_t random_between(uint64_t *state, uint64_t x, uint64_t y) {
    if (x == y)
        return x;
    uint64_t r = random_xorshift64(*state);
    return x + (r % (y - x + 1));
}

__forceinline__ __device__ void
dummy_work_flop(uint32_t n_flop) {
    float x = 0.5f + threadIdx.x;
    float y = 1.0001f + threadIdx.x;
    float z = 0.0f;

    uint32_t n_iters = n_flop / 4;

    for (uint64_t i = 0; i < n_iters; ++i) {
        z = z + x;        // 1 FLOP
        z = z * y;        // 1 FLOP
        x = x * 1.00001f; // 1 FLOP
        y = x + z;        // 1 FLOP
    }

    volatile float sink = z;
}

__device__ int my_strcmp(const char *s1, const char *s2) {
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(const unsigned char *)s1 - *(const unsigned char *)s2;
}

__device__ __forceinline__ uint32_t
get_smid2(void) {
    uint32_t ret;
    asm("mov.u32 %0, %smid;"
        : "=r"(ret));
    return ret;
}

#define sleep_wait_assert_printf(ns, max_ns, wait_cnt, max_wait_cnt, ...) \
    {                                                                     \
        __nanosleep(ns);                                                  \
        if (ns < max_ns)                                                  \
            ns *= 2;                                                      \
        ++wait_cnt;                                                       \
        if (wait_cnt > max_wait_cnt) {                                    \
            printf(__VA_ARGS__);                                          \
            force_assert(false);                                          \
        }                                                                 \
    }

#pragma once

#include <cuda.h>

__forceinline__ __device__ uint32_t
fnv_hash(const uint8_t *key, uint32_t len) {
    // uint32_t hash = 0;
    uint32_t hash = 0x811c9dc5;
    for (uint32_t i = 0; i < len; ++i) {
        hash = hash * 0x01000193;
        hash = hash ^ key[i];
    }
    return hash;
}
__forceinline__ __device__ uint32_t
fnv_hash_len4(const uint8_t *key) {
    uint32_t hash = 0x811c9dc5;
    // for (uint32_t i = 0; i < 4; ++i) {
    //     hash = hash * 0x01000193;
    //     hash = hash ^ key[i];
    // }
    hash = hash * 0x01000193;
    hash = hash ^ key[0];
    hash = hash * 0x01000193;
    hash = hash ^ key[1];
    hash = hash * 0x01000193;
    hash = hash ^ key[2];
    hash = hash * 0x01000193;
    hash = hash ^ key[3];
    return hash;
}
__forceinline__ __device__ uint32_t
prime_hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

__forceinline__ __device__ uint32_t
simple_hash(const uint8_t *key, uint32_t len) {
    return fnv_hash(key, len);
}
__forceinline__ __device__ uint32_t
simple_hash_len4(const uint8_t *key) {
    return fnv_hash_len4(key);
}

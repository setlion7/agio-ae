#pragma once
#include <cublas_v2.h>
#include <cuda.h>
#include <nvml.h>

#include <cstdio>

#ifndef G_DEF_CUDAERRCHK
#define G_DEF_CUDAERRCHK
#define cudaErrChk(err)                                                                                          \
    do {                                                                                                         \
        cudaError_t err_ = (err);                                                                                \
        if (err_ != cudaSuccess) {                                                                               \
            fprintf(stderr, "CUDA error %d at %s:%d, %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(err_);                                                                                          \
        }                                                                                                        \
    } while (0)
#endif

#define cuErrChk(err)                                                                     \
    do {                                                                                  \
        CUresult err_ = (err);                                                            \
        if (err_ != CUDA_SUCCESS) {                                                       \
            const char *str;                                                              \
            cuGetErrorString(err_, &str);                                                 \
            fprintf(stderr, "CU error %d at %s:%d, %s\n", err_, __FILE__, __LINE__, str); \
            exit(err_);                                                                   \
        }                                                                                 \
    } while (0)

#define nvmlErrChk(err)                                                                                       \
    do {                                                                                                      \
        nvmlReturn_t err_ = (err);                                                                            \
        if (err_ != NVML_SUCCESS) {                                                                           \
            fprintf(stderr, "NVML error %d at %s:%d, %s\n", err_, __FILE__, __LINE__, nvmlErrorString(err_)); \
            exit(err_);                                                                                       \
        }                                                                                                     \
    } while (0)

#define cublasErrChk(err)                                                            \
    do {                                                                             \
        cublasStatus_t err_ = (err);                                                 \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "CUBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            exit(err_);                                                              \
        }                                                                            \
    } while (0)

#define cufileErrChk(err_in)                                                \
    do {                                                                    \
        CUfileError_t status = (err_in);                                    \
        if (status.err != CU_FILE_SUCCESS) {                                \
            fprintf(stderr, "CUFILE error at %s:%d, %s\n", __FILE__, __LINE__, CUFILE_ERRSTR(status.err)); \
            exit(status.err);                                               \
        }                                                                   \
    } while (0)

#pragma once

#include <cstdlib>
#include "cuda.h"

#include "fassert.cuh"

void checkNVML(bool print, nvmlDevice_t nvmldev, unsigned int expected_gpu_clock_mhz) {
    unsigned int current_sm_clock = 0;
    unsigned int current_mem_clock = 0;
    nvmlErrChk(nvmlDeviceGetClockInfo(nvmldev, NVML_CLOCK_SM, &current_sm_clock));
    nvmlErrChk(nvmlDeviceGetClockInfo(nvmldev, NVML_CLOCK_MEM, &current_mem_clock));

    nvmlBAR1Memory_t bar1;
    nvmlErrChk(nvmlDeviceGetBAR1MemoryInfo(nvmldev, &bar1));
    if (print) {
        printf("%s: NVML_CLOCK_SM: %u, NVML_CLOCK_MEM: %u\n", __func__, current_sm_clock, current_mem_clock);
        printf("%s: BAR1: Total: %llu MiB, Free: %llu MiB\n", __func__, bar1.bar1Total / 1024 / 1024, bar1.bar1Free / 1024 / 1024);
    }

    force_assert(bar1.bar1Total >= 1024 * 1024 * 1024);
    if (current_sm_clock != expected_gpu_clock_mhz) {
        fprintf(stderr, "%s: invalid clock. current: %u, expected: %u (MHz)\n", __func__, current_sm_clock, expected_gpu_clock_mhz);
        exit(EXIT_FAILURE);
    }
}

void checkProp(bool print, cudaDeviceProp prop) {
    if (print) {
        printf("%s: number of multiprocessors: %d\n", __func__, prop.multiProcessorCount);
        printf("%s: max threads / MP: %d\n", __func__, prop.maxThreadsPerMultiProcessor);
        printf("%s: max threads / block: %d\n", __func__, prop.maxThreadsPerBlock);
        printf("%s: computePreemptionSupported: %d\n", __func__, prop.computePreemptionSupported);
        printf("%s: l2CacheSize: %d bytes\n", __func__, prop.l2CacheSize);
        printf("%s: unifiedAddressing: %d\n", __func__, prop.unifiedAddressing);
        printf("%s: concurrentKernels: %d\n", __func__, prop.concurrentKernels);
    }

    force_assert(prop.unifiedAddressing == 1);
    force_assert(prop.concurrentKernels == 1);
}

void gpuProp(int cudaDevice, unsigned int expected_gpu_clock_mhz) {
    nvmlDevice_t nvmldev;
    nvmlErrChk(nvmlInit_v2());
    nvmlErrChk(nvmlDeviceGetHandleByIndex_v2(cudaDevice, &nvmldev));

    cudaDeviceProp prop;
    cudaErrChk(cudaGetDeviceProperties(&prop, cudaDevice));
    checkNVML(false, nvmldev, expected_gpu_clock_mhz);
    checkProp(false, prop);

    size_t cuda_stack_size = 1024 * 8;
    size_t cuda_printf_size;
    size_t launch_limit;
    cudaErrChk(cudaDeviceSetLimit(cudaLimitStackSize, cuda_stack_size));
    cudaErrChk(cudaDeviceGetLimit(&cuda_stack_size, cudaLimitStackSize));
    cudaErrChk(cudaDeviceGetLimit(&cuda_printf_size, cudaLimitPrintfFifoSize));
    cudaErrChk(cudaDeviceGetLimit(&launch_limit, cudaLimitDevRuntimePendingLaunchCount));
    printf("cuda_stack_size: %lu, cuda_printf_size: %lu, launch_limit: %lu\n",
           cuda_stack_size, cuda_printf_size, launch_limit);

    // Kernel-side malloc() global memory heap size
    size_t kernel_malloc_heap_size = 0;
    cudaErrChk(cudaDeviceGetLimit(&kernel_malloc_heap_size, cudaLimitMallocHeapSize));
    // printf("%s: malloc_heap_size: %lu\n", __func__, kernel_malloc_heap_size);
    // kernel_malloc_heap_size = 1024 * 1024 * 1024 * 1UL;
    // cudaErrChk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, kernel_malloc_heap_size));
    // cudaErrChk(cudaDeviceGetLimit(&kernel_malloc_heap_size, cudaLimitMallocHeapSize));
    // printf("%s: malloc_heap_size: %lu\n", __func__, kernel_malloc_heap_size);

    int leastPriority = 0;
    int greatestPriority = 0;
    cudaErrChk(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    // printf("%s: stream priority: %d ~ %d\n", __func__, leastPriority, greatestPriority);
}

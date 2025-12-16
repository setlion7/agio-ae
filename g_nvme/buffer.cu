#include <iostream>
#include <memory>

#include "cuda.h"
#include "buffer.cuh"

#include "gpuErrChk.h"

// Create DMA-able memory on the CPU memory
// DmaPtr createDma_host(const struct gnvme_ctrl_t *ctrl, size_t size)
// {
// }

// Create DMA-able memory on the CUDA device memory
std::shared_ptr<nvm_dma_t> createDma_cuda(const nvm_ctrl_t *ctrl, size_t size)
{
    nvm_dma_t *dma = nullptr;
    void *bufferPtr = nullptr;
    void *original = nullptr;

    // Align memory
    size = size + 64 * 1024;
    cudaErrChk(cudaMalloc(&original, size));
    bufferPtr = original;
    bufferPtr = (void *)((((uint64_t)bufferPtr) + (64 * 1024)) & 0xffffffffff0000);

    int status = nvm_dma_map_device(&dma, ctrl, bufferPtr, size);
    if (!nvm_ok(status))
    {
        throw std::string("Failed to map device memory: ") + nvm_strerror(status);
    }

    // cudaErrChk(cudaMemset(bufferPtr, 0, size));
    cudaErrChk(cudaMemset(original, 0, size));
    dma->vaddr = bufferPtr;

    return std::shared_ptr<nvm_dma_t>(dma, [bufferPtr, original](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(original);
        // std::cout << "Deleting DMA_cuda\n";
    });
}

#pragma once

#include <memory>

// BaM
#include "nvm_types.h"
#include "nvm_dma.h"
#include "nvm_error.h"

std::shared_ptr<nvm_dma_t> createDma_cuda(const nvm_ctrl_t *ctrl, size_t size);

#pragma once

#include <string>

struct bam_array_in {
    uint32_t nvme_ctrl_id;
    uint64_t bam_array_nvme_start_byte;
    uint64_t bam_array_size_bytes;
    // char bam_array_name[512];
    std::string bam_array_name;
};

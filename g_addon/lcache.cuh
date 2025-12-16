#pragma once

#include <exception>
#include <vector>
#include <iostream>
#include <string>

#include "helper_headers/json.hpp"
#include "helper_headers/fassert.cuh"

#include "export.cuh"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wall"
#endif

#pragma nv_diagnostic push
#pragma nv_diag_suppress 177
// variable "ns" was declared but never referenced
#pragma nv_diag_suppress 550
// variable "pc_pos" was set but never used
#pragma nv_diag_suppress 546
// transfer of control bypasses initialization of

#include "page_cache.h"

#pragma nv_diagnostic pop
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

typedef uint32_t bam_t;
#define BAM_MAX_ARRAYS_PER_CTRL 4

// from page_cache.h
template <typename T>
void print_reset_stats_json_new(array_d_t<T> &adt, nlohmann::json &out_js, std::string arr_name) {
    std::vector<range_d_t<T>> rdt(adt.n_ranges);
    // range_d_t<T>* rdt = new range_d_t<T>[adt.n_ranges];
    cuda_err_chk(cudaMemcpy(rdt.data(), adt.d_ranges, adt.n_ranges * sizeof(range_d_t<T>), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < adt.n_ranges; i++) {

        std::string range_id(std::to_string(i));
        out_js["cache_stats"][arr_name][range_id].emplace("access_cnt", rdt[i].access_cnt.load());
        out_js["cache_stats"][arr_name][range_id].emplace("miss_cnt", rdt[i].miss_cnt.load());
        out_js["cache_stats"][arr_name][range_id].emplace("hit_cnt", rdt[i].hit_cnt.load());
        out_js["cache_stats"][arr_name][range_id].emplace("miss_rate", (float)rdt[i].miss_cnt.load() / rdt[i].access_cnt.load());
        out_js["cache_stats"][arr_name][range_id].emplace("hit_rate", (float)rdt[i].hit_cnt.load() / rdt[i].access_cnt.load());
        std::cout << std::dec << "#READ IOs: " << rdt[i].read_io_cnt
                  << "\t#Accesses:" << rdt[i].access_cnt
                  << "\t#Misses:" << rdt[i].miss_cnt
                  << "\tMiss Rate:" << ((float)rdt[i].miss_cnt / rdt[i].access_cnt)
                  << "\t#Hits: " << rdt[i].hit_cnt
                  << "\tHit Rate:" << ((float)rdt[i].hit_cnt / rdt[i].access_cnt)
                  << "\tCLSize:" << rdt[i].page_size
                  << std::endl;
        std::cout << "*********************************" << std::endl;
        rdt[i].read_io_cnt = 0;
        rdt[i].access_cnt = 0;
        rdt[i].miss_cnt = 0;
        rdt[i].hit_cnt = 0;
    }
    cuda_err_chk(cudaMemcpy(adt.d_ranges, rdt.data(), adt.n_ranges * sizeof(range_d_t<T>), cudaMemcpyHostToDevice));
}

// one struct per array
template <typename T>
class range_array {
    public:
    std::vector<range_t<T> *> h_vec_range;
    array_t<T> *h_array;
    uint64_t range_nvme_start_byte;
    uint64_t range_nvme_end_byte;
    std::string name;

    void range_array_init(int cudaDevice, page_cache_t *bam_h_pc,
                              uint64_t nvme_start_byte, uint64_t nvme_num_bytes, std::string name_in) {
        force_assert(nvme_num_bytes > 0);
        force_assert(nvme_num_bytes % sizeof(bam_t) == 0);
        uint64_t num_elems = nvme_num_bytes / sizeof(bam_t);
        uint64_t page_size = bam_h_pc->pdt.page_size;
        uint64_t page_count = nvme_num_bytes / page_size;

        uint64_t page_start_loc = nvme_start_byte / page_size;

        h_vec_range.resize(1);
        // BaM array
        // index_start, count, page_start, page_count, page_start_offset, page_size, page_cache, cudaDevice
        h_vec_range[0] = new range_t<bam_t>((uint64_t)0, num_elems, page_start_loc, (uint64_t)page_count,
                                            (uint64_t)0, (uint64_t)page_size, bam_h_pc, cudaDevice);
        h_array = new array_t<bam_t>(num_elems, 77777777, h_vec_range, cudaDevice);
        range_nvme_start_byte = nvme_start_byte;
        range_nvme_end_byte = nvme_start_byte + nvme_num_bytes - 1;
        name = name_in;
        printf("%s: adding range/array: %s, range_dptr: %p, start byte: %lu, num bytes: %lu\n",
               __func__, name.c_str(), h_vec_range[0]->d_range_ptr, nvme_start_byte, nvme_num_bytes);
    }
    void range_array_fini() {
        delete h_array;
        delete h_vec_range[0];
    }
};

// one struct per ctrl
template <typename T>
struct cache_ctrl {
    page_cache_t *h_page_cache;
    std::vector<struct range_array<T>> ra_vec;
};

// one struct per ctrl, on device
struct __align__(32) bam_array_d {
    uint32_t num_bam_array;
    range_d_t<bam_t> *bam_range[BAM_MAX_ARRAYS_PER_CTRL];
    array_d_t<bam_t> *bam_array[BAM_MAX_ARRAYS_PER_CTRL];
    uint64_t bam_array_nvme_start_byte[BAM_MAX_ARRAYS_PER_CTRL];
    uint64_t bam_array_nvme_end_byte[BAM_MAX_ARRAYS_PER_CTRL];
    // void *bam_pc_base_addr;
    // uint64_t bam_pc_page_size;
};

class __align__(32) lcache {
    public:
    struct bam_array_d *b_arr;

    __device__ void lcache_init_d() {
    }
    __device__ void lcache_fini_d() {
    }

    __host__ void lcache_init_h(std::vector<struct cache_ctrl<bam_t>> &cache_ctrl_vec) {
        static_assert(std::is_standard_layout_v<lcache> == true);
        static_assert(std::is_trivially_copyable_v<lcache> == true);

        std::vector<struct bam_array_d> bam_array_d;
        bam_array_d.resize(cache_ctrl_vec.size());

        for (uint32_t ctrl_id = 0; ctrl_id < cache_ctrl_vec.size(); ++ctrl_id) {
            struct cache_ctrl<bam_t> &cc = cache_ctrl_vec[ctrl_id];
            struct bam_array_d arrd{};
            arrd.num_bam_array = cc.ra_vec.size();
            force_assert(arrd.num_bam_array < BAM_MAX_ARRAYS_PER_CTRL);
            for (uint32_t i = 0; i < arrd.num_bam_array; ++i) {
                // arrd.bam_range[i] = cc.ra_vec[i].h_vec_range[0]->d_range_ptr;
                arrd.bam_range[i] = cc.ra_vec[i].h_array->adt.d_ranges;
                force_assert(cc.ra_vec[i].h_array->adt.n_ranges == 1);
                arrd.bam_array[i] = cc.ra_vec[i].h_array->d_array_ptr;
                arrd.bam_array_nvme_start_byte[i] = cc.ra_vec[i].range_nvme_start_byte;
                arrd.bam_array_nvme_end_byte[i] = cc.ra_vec[i].range_nvme_end_byte;
            }
            // arrd.bam_pc_base_addr = cc.h_page_cache->pdt.base_addr;
            // arrd.bam_pc_page_size = cc.h_page_cache->pdt.page_size;
            bam_array_d[ctrl_id] = arrd;
            // printf("%s: ctrl: %u, range_size_GiB: %u\n", __func__, i, range_size_GiB);
        }

        std::remove_pointer_t<decltype(this)> lc;
        uint64_t arr_len = bam_array_d.size();
        cudaErrChk(cudaMalloc(&(lc.b_arr), sizeof(*lc.b_arr) * arr_len));
        cudaErrChk(cudaMemcpy(lc.b_arr, bam_array_d.data(), sizeof(*lc.b_arr) * arr_len, cudaMemcpyHostToDevice));
        memcpy(this, &lc, sizeof(lc));
    }
    __host__ void lcache_fini_h() {
        std::remove_pointer_t<decltype(this)> lc;
        memcpy(reinterpret_cast<char *>(&lc), this, sizeof(lc));
        cudaErrChk(cudaFree(lc.b_arr));
    }
};

class __align__(32) lcache_w {
    public:
    class lcache lcache_h;
    class lcache *lcache_dp;

    uint32_t range_mode;
    uint32_t range_size_GiB;

    int cudaDevice;

    uint32_t num_ctrls;
    bool bam_page_cache_single;
    std::vector<struct cache_ctrl<bam_t>> cache_ctrl_vec;

    __host__ void
    lcache_w_init_h(std::vector<Controller *> ctrls, std::vector<struct bam_array_in> bam_in_vec, const nlohmann::json in_js) {
        static_assert(std::is_standard_layout_v<lcache_w> == true);

        try {
            range_mode = in_js.at("range_mode");
            if (range_mode == 1)
                range_size_GiB = in_js.at("range_size_GiB");
        } catch (const std::exception &e) {
            std::cerr << e.what() << '\n';
            force_assert(false);
        }
        cudaDevice = 0;

        size_t page_size = 4096;
        size_t pc_size_bytes = 0;
        try {
            pc_size_bytes = in_js.at("cache_size_bytes");
        } catch (const std::exception &e) {
            // try with different key
            try {
                pc_size_bytes = in_js.at("cache_size_GiB");
                pc_size_bytes = pc_size_bytes * 1024ULL * 1024ULL * 1024ULL;
            } catch (const std::exception &e) {
                std::cerr << e.what() << '\n';
                force_assert(false);
            }
        }
        force_assert(pc_size_bytes % page_size == 0);
        size_t pc_n_pages = pc_size_bytes / page_size;

        num_ctrls = (uint32_t)ctrls.size();
        cache_ctrl_vec.resize(num_ctrls);

        bam_page_cache_single = true;
        cache_ctrl_vec[0].h_page_cache = new page_cache_t(page_size, pc_n_pages, cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
        if (!bam_page_cache_single) {
            for (uint32_t i = 1; i < num_ctrls; ++i) {
                // page size, number of pages, cudaDevice, controller, max ranges, controller vector
                cache_ctrl_vec[i].h_page_cache = new page_cache_t(page_size, pc_n_pages, cudaDevice, ctrls[i][0], (uint64_t)64, ctrls);
            }
        }

        if (range_mode == 1) {
            // use single range from NVMe byte 0
            for (uint32_t i = 0; i < num_ctrls; ++i) {
                uint64_t range_size_bytes = 0;
                uint64_t namespace_size_bytes = ctrls[i]->ns.size * ctrls[i]->ns.lba_data_size;
                if (range_size_GiB == 0)
                    range_size_bytes = namespace_size_bytes;
                else
                    range_size_bytes = (uint64_t)range_size_GiB * 1024UL * 1024UL * 1024UL;

                class range_array<bam_t> ds{};
                if (bam_page_cache_single) {
                    ds.range_array_init(cudaDevice, cache_ctrl_vec[0].h_page_cache,
                                            0, range_size_bytes, "default");
                } else {
                    ds.range_array_init(cudaDevice, cache_ctrl_vec[i].h_page_cache,
                                            0, range_size_bytes, "default");
                }
                cache_ctrl_vec[i].ra_vec.push_back(ds);
            }
        } else if (range_mode == 2) {
            // use range from work
            for (uint32_t i = 0; i < bam_in_vec.size(); ++i) {
                uint32_t nvme_ctrl_id = bam_in_vec[i].nvme_ctrl_id;

                class range_array<bam_t> ds{};
                if (bam_page_cache_single) {
                    ds.range_array_init(cudaDevice, cache_ctrl_vec[0].h_page_cache,
                                            bam_in_vec[i].bam_array_nvme_start_byte, bam_in_vec[i].bam_array_size_bytes, bam_in_vec[i].bam_array_name);
                } else {
                    ds.range_array_init(cudaDevice, cache_ctrl_vec[nvme_ctrl_id].h_page_cache,
                                            bam_in_vec[i].bam_array_nvme_start_byte, bam_in_vec[i].bam_array_size_bytes, bam_in_vec[i].bam_array_name);
                }
                cache_ctrl_vec[nvme_ctrl_id].ra_vec.push_back(ds);
            }
        } else {
            force_assert(false);
        }

        lcache_h.lcache_init_h(cache_ctrl_vec);
        cudaErrChk(cudaMalloc(&(lcache_dp), sizeof(*lcache_dp)));
        cudaErrChk(cudaMemcpy(lcache_dp, &lcache_h, sizeof(*lcache_dp), cudaMemcpyHostToDevice));

        printf("%s: cache init complete. num_ctrls: %u, PC size: %.3lf MiB\n",
               __func__, num_ctrls, (double)pc_size_bytes / 1024.0 / 1024.0);
    }
    __host__ void
    lcache_w_fini_h() {
        cudaErrChk(cudaFree(lcache_dp));
        lcache_h.lcache_fini_h();

        for (uint32_t i = 0; i < num_ctrls; ++i) {
            for (uint32_t j = 0; j < cache_ctrl_vec[i].ra_vec.size(); ++j) {
                cache_ctrl_vec[i].ra_vec[j].range_array_fini();
            }
        }

        if (!bam_page_cache_single) {
            for (uint32_t i = 1; i < num_ctrls; ++i) {
                delete cache_ctrl_vec[i].h_page_cache;
            }
        }
        delete cache_ctrl_vec[0].h_page_cache;
    }

    __host__ void
    stats_out(nlohmann::json &out_js);
};

__host__ void
lcache_w::stats_out(nlohmann::json &out_js) {
    for (uint32_t i = 0; i < num_ctrls; ++i) {
        struct cache_ctrl<bam_t> &cc = cache_ctrl_vec[i];
        for (uint32_t j = 0; j < cc.ra_vec.size(); ++j) {
            // cc.ra_vec[j].h_array->print_reset_stats_json(out_js, cc.ra_vec[j].name);
            print_reset_stats_json_new<bam_t>((cc.ra_vec[j].h_array->adt), out_js, cc.ra_vec[j].name);
        }
    }
}

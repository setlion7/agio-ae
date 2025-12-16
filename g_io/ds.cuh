#pragma once

#include <cuda/atomic>

#include "helper_headers/fassert.cuh"
#include "helper_headers/gpuErrChk.h"

#include "dynamic.cuh"
#include "comm.cuh"
#include "macro.h"
#include "types.cuh"

#include "memalloc.cuh"
#include "work_ds.cuh"

#include "lcache.cuh"
#include "ctrl.h"

__device__ __align__(32) class gio *gp_g;

typedef uint32_t lstage_status_t;

struct __align__(4) lstage_entry {
    cuda::atomic<uint32_t, cuda::thread_scope_device> consumed;
    uint64_t command_id;
};

class __align__(32) lstage {
    public:
    struct padded_atomic_status<lstage_status_t> *entry_status;
    struct lstage_entry *entry_buffer;
    cuda::atomic<uint32_t, cuda::thread_scope_device> next;

    __device__ void lstage_init_d() {
        next.store(0);
    }
    __device__ void lstage_fini_d() {
    }

    __host__ void lstage_init_h(struct padded_atomic_status<lstage_status_t> *status_dp_in,
                                struct lstage_entry *buffer_dp_in) {
        static_assert(std::is_standard_layout_v<lstage> == true);
        static_assert(std::is_trivially_copyable_v<lstage> == true);

        std::remove_pointer_t<decltype(this)> cq;
        cq.entry_status = status_dp_in;
        cq.entry_buffer = buffer_dp_in;
        memcpy(this, &cq, sizeof(*this));
    }
    __host__ void lstage_fini_h() {
    }

    __forceinline__ __device__ void post_entry(uint64_t command_id) {
        uint32_t new_idx = next.fetch_add(1, cuda::memory_order_relaxed) % n_stage_entries;
        entry_buffer[new_idx].command_id = command_id;
        entry_buffer[new_idx].consumed.store((lstage_status_t)(ENTRY_PROCESSED ^ command_id), cuda::memory_order_release);
    }

    __forceinline__ __device__ int32_t _check_block_64(const uint64_t cid) {
        uint32_t ns = 8;
        uint32_t wait_cnt = 0;
        while (true) {
            // loop within the entire bucket
            for (uint32_t i = 0; i < n_stage_entries; ++i) {
                if (entry_buffer[i].consumed.load(cuda::memory_order_acquire) == (lstage_status_t)(ENTRY_PROCESSED ^ cid)) {
                    if (entry_buffer[i].command_id != cid) {
                        continue;
                    }
                    entry_buffer[i].consumed.store((lstage_status_t)ENTRY_EMPTY, cuda::memory_order_relaxed);
                    return 0;
                }
            }
            // sleep_wait_assert_printf(ns, 512, wait_cnt, 1024 * 1024, "cid: %lu\n", cid);
            __nanosleep(ns);
            if (ns < 512)
                ns *= 2;
            wait_cnt++;
            if (wait_cnt > 16)
                return -1;
        }
    }
};

class __align__(4096) gio {
    public:
    // addon
    class memalloc *mptr;
    class lcache *cp;
    class warp_smid *wsp;

    class dynamic dy;

    uint32_t end_value; // exit value

    cuda::atomic<uint32_t, cuda::thread_scope_device> user_start_num_warps;
    cuda::atomic<uint32_t, cuda::thread_scope_device> user_done_num_warps;
    cuda::atomic<uint32_t, cuda::thread_scope_device> rt_start_num_warps;
    cuda::atomic<uint32_t, cuda::thread_scope_device> rt_running;

    uint32_t total_num_lib_warps;
    uint32_t total_num_user_warps;

    class lstage *stage;
    uint32_t n_stage;

    // Queues
    struct s_queue __align__(128) * s_queue;
    uint32_t n_sq;
    uint32_t sq_threshold;
    uint32_t wq_threshold;
    gpua_tag_t tag_enabled;
    uint32_t n_queues_per_rtsm;
    uint32_t n_wq;

    // head poller warp
    cuda::atomic<int32_t, cuda::thread_scope_device> smid_assign;
    cuda::atomic<int32_t, cuda::thread_scope_device> *smid_table;
    cuda::atomic<int32_t, cuda::thread_scope_device> __align__(64) * s_queue_poller;

    struct wait_ds __align__(128) wait_ds;

    class Controller **ctrl_dptrs;
    uint32_t num_ctrls;

    __device__ void gio_init_d() {
        gp_g = this;

        dy.dynamic_init_d(n_sq, n_sq, sq_threshold, wq_threshold);
        tag_enabled = 0xFF;

        user_start_num_warps.store(0);
        user_done_num_warps.store(0);
        rt_start_num_warps.store(0);
        rt_running.store(0);

        end_value = 2;

        smid_assign.store(0);
        for (uint32_t i = 0; i < 108; ++i) {
            smid_table[i].store(-1);
        }
        // Queues
        for (uint32_t i = 0; i < n_sq; ++i) {
            new (&(s_queue[i])) struct s_queue;
            s_queue[i].s_queue_init_d(i, sq_threshold);
            s_queue_poller[i].store(-1);
        }
        for (uint32_t i = 0; i < n_stage; ++i) {
            new (&(stage[i])) class lstage;
            stage[i].lstage_init_d();
        }
        new (&wait_ds) struct wait_ds;
        wait_ds.wait_ds_init_d(wq_threshold);
    };
    __device__ void gio_fini_d() {
        for (uint32_t i = 0; i < n_sq; ++i) {
            s_queue[i].s_queue_fini_d();
            s_queue[i].~s_queue();
        }
        for (uint32_t i = 0; i < n_stage; ++i) {
            stage[i].lstage_fini_d();
            stage[i].~lstage();
        }
        wait_ds.wait_ds_fini_d();
        wait_ds.~wait_ds();

        dy.dynamic_fini_d();
    };

    __host__ void gio_init_h(std::vector<std::tuple<uint64_t, uint64_t, void *>> &prp_vec,
                             std::vector<class Controller *> &ctrl_dptrs_in) {
        static_assert(std::is_standard_layout_v<gio> == true);
        static_assert(std::is_trivially_copyable_v<gio> == true);

        end_value = 9;

        decltype(dy) dy_temp_h;
        dy_temp_h.dynamic_init_h();
        cudaErrChk(cudaMemcpy(&dy, &dy_temp_h, sizeof(dy), cudaMemcpyHostToDevice));

        decltype(wait_ds) wait_ds_temp_h;
        wait_ds_temp_h.wait_ds_init_h(n_wq, n_wq_entries);
        cudaErrChk(cudaMemcpy(&wait_ds, &wait_ds_temp_h, sizeof(wait_ds), cudaMemcpyHostToDevice));

        cudaErrChk(cudaMalloc(&s_queue_poller, sizeof(cuda::atomic<int32_t, cuda::thread_scope_device>) * n_sq));

        // sq
        std::vector<struct s_queue> s_queue_temp_h(n_sq);
        for (uint32_t i = 0; i < n_sq; ++i) {
            s_queue_temp_h[i].s_queue_init(n_sq_entries, std::get<0>(prp_vec[i]), std::get<1>(prp_vec[i]), std::get<2>(prp_vec[i]));
        }
        cudaErrChk(cudaMalloc(&s_queue, sizeof(struct s_queue) * n_sq));
        cudaErrChk(cudaMemcpy(s_queue, s_queue_temp_h.data(), sizeof(struct s_queue) * n_sq, cudaMemcpyHostToDevice));

        // lstage
        std::vector<struct padded_atomic_status<lstage_status_t>> temp1_h(n_stage_entries * n_stage);
        for (size_t i = 0; i < temp1_h.size(); ++i) {
            temp1_h[i].value = ENTRY_EMPTY;
        }
        struct padded_atomic_status<lstage_status_t> *temp1 = nullptr;
        cudaErrChk(cudaMalloc(&temp1, sizeof(*temp1) * n_stage_entries * n_stage));
        cudaErrChk(cudaMemcpy(temp1, temp1_h.data(), sizeof(*temp1) * temp1_h.size(), cudaMemcpyHostToDevice));

        std::vector<struct lstage_entry> temp2_h(n_stage_entries * n_stage);
        for (size_t i = 0; i < temp2_h.size(); ++i) {
            temp2_h[i].command_id = 0;
            temp2_h[i].consumed = ENTRY_EMPTY;
        }
        struct lstage_entry *temp2 = nullptr;
        cudaErrChk(cudaMalloc(&temp2, sizeof(*temp2) * n_stage_entries * n_stage));
        cudaErrChk(cudaMemcpy(temp2, temp2_h.data(), sizeof(*temp2) * temp2_h.size(), cudaMemcpyHostToDevice));

        std::vector<class lstage> c_queue_h(n_stage);
        for (uint32_t i = 0; i < n_stage; ++i) {
            struct padded_atomic_status<lstage_status_t> *t1 = temp1 + (n_stage_entries * i);
            struct lstage_entry *t2 = temp2 + (n_stage_entries * i);
            c_queue_h[i].lstage_init_h(t1, t2);
        }
        cudaErrChk(cudaMalloc(&stage, sizeof(*stage) * n_stage));
        cudaErrChk(cudaMemcpy(stage, c_queue_h.data(), sizeof(c_queue_h[0]) * n_stage, cudaMemcpyHostToDevice));

        // NVMe controllers
        num_ctrls = ctrl_dptrs_in.size();
        cudaErrChk(cudaMalloc(&ctrl_dptrs, sizeof(class Controller *) * ctrl_dptrs_in.size()));
        for (uint32_t i = 0; i < ctrl_dptrs_in.size(); ++i) {
            cudaErrChk(cudaMemcpy(&(ctrl_dptrs[i]), &(ctrl_dptrs_in[i]->d_ctrl_ptr), sizeof(class Controller *), cudaMemcpyHostToDevice));
        }

        cudaErrChk(cudaMalloc(&smid_table, sizeof(cuda::atomic<int32_t, cuda::thread_scope_device>) * 108)); // FIXME
    }
    __host__ void gio_fini_h() {
        cudaErrChk(cudaFree(ctrl_dptrs));

        cudaErrChk(cudaFree(smid_table));

        // lstage
        std::vector<class lstage> c_queue_temp_h(n_stage);
        cudaErrChk(cudaMemcpy(c_queue_temp_h.data(), stage, sizeof(c_queue_temp_h[0]) * n_stage, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n_stage; ++i) {
            c_queue_temp_h[i].lstage_fini_h();
        }
        cudaErrChk(cudaFree(c_queue_temp_h[0].entry_status));
        cudaErrChk(cudaFree(c_queue_temp_h[0].entry_buffer));
        cudaErrChk(cudaFree(stage));

        std::vector<struct s_queue> s_queue_temp_h(n_sq);
        cudaErrChk(cudaMemcpy(s_queue_temp_h.data(), s_queue, sizeof(s_queue_temp_h[0]) * n_sq, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n_sq; ++i) {
            s_queue_temp_h[i].s_queue_fini();
        }
        cudaErrChk(cudaFree(s_queue));

        cudaErrChk(cudaFree(s_queue_poller));

        decltype(wait_ds) wait_ds_temp_h;
        cudaErrChk(cudaMemcpy(&wait_ds_temp_h, &wait_ds, sizeof(wait_ds), cudaMemcpyDeviceToHost));
        wait_ds_temp_h.wait_ds_fini_h();

        decltype(dy) dy_temp_h;
        cudaErrChk(cudaMemcpy(&dy_temp_h, &dy, sizeof(dy), cudaMemcpyDeviceToHost));
        dy_temp_h.dynamic_fini_h();
    }

    __forceinline__ __device__ void runtime_main(const uint32_t tid, const uint32_t groupId);
    __forceinline__ __device__ void process_io(struct s_queue *local_sq, sq_ptr_t sq_entry_idx, gpua_tag_t tag);
    __forceinline__ __device__ void notify_io(struct s_queue *local_sq, sq_ptr_t sq_entry_idx, gpua_tag_t tag);
    __forceinline__ __device__ void shutdown();

    __host__ void print_stats(float main_cuda_event_ms, nlohmann::json &out_js);
};

__host__ void
gio::print_stats(float main_cuda_event_ms, nlohmann::json &out_js) {
}

class gio_w {
    public:
    class gio *gio_dp;
    class gio gio_h;

    // DMA shared ptr
    std::vector<std::shared_ptr<nvm_dma_t>> nvm_dma_shared_ptr;

    uint32_t prp_dma_num_pages_temp;

    void gio_w_init_h(std::vector<class Controller *> ctrl_dptrs_in) {
        std::vector<std::tuple<uint64_t, uint64_t, void *>> prp_vec;
        for (uint32_t i = 0; i < gio_h.n_sq; ++i) {
            // PRP memory
            uint64_t prp_dma_num_pages = prp_dma_num_pages_temp;

            nvm_dma_shared_ptr.push_back(createDma_cuda(ctrl_dptrs_in[0]->ctrl, prp_dma_num_pages * 4096UL));
            force_assert(nvm_dma_shared_ptr[i].get()->contiguous == true); // DMA addresses must be contiguous
            force_assert(nvm_dma_shared_ptr[i].get()->page_size == 4096);

            uint64_t prp_dma_buffer_ioaddr = nvm_dma_shared_ptr[i].get()->ioaddrs[0];
            void *prp_dma_buffer_dptr = nvm_dma_shared_ptr[i].get()->vaddr;
            prp_vec.push_back(std::make_tuple(prp_dma_num_pages, prp_dma_buffer_ioaddr, prp_dma_buffer_dptr));
        }

        gio_h.gio_init_h(prp_vec, ctrl_dptrs_in);
        cudaErrChk(cudaMalloc(&gio_dp, sizeof(*gio_dp)));
        cudaErrChk(cudaMemcpy(gio_dp, &gio_h, sizeof(*gio_dp), cudaMemcpyHostToDevice));
    }
    void gio_w_fini_h() {
        gio_h.gio_fini_h();
        cudaErrChk(cudaFree(gio_dp));

        for (uint32_t i = 0; i < gio_h.n_sq; ++i) {
            // Delete DMA memory
            nvm_dma_shared_ptr[i].reset();
        }
    }

    void copy_options(const nlohmann::json g_cuda_js, const nlohmann::json in_js);
    void print_stats(nlohmann::json &out_js);

    uint32_t get_total_n_lib_warps() {
        return gio_h.total_num_lib_warps;
    }
    uint32_t get_total_n_user_warps() {
        return gio_h.total_num_user_warps;
    }
};

void gio_w::copy_options(const nlohmann::json g_cuda_js, nlohmann::json js) {
    uint32_t max_rt_SMs = g_cuda_js.at("green_rt_n_groups").template get<uint32_t>() *
                          g_cuda_js.at("green_group_size").template get<uint32_t>();
    nlohmann::json sq_js = js.at("n_sq");
    if ((sq_js.is_string()) && (sq_js.template get<std::string>().compare("max") == 0)) {
        uint32_t temp = max_rt_SMs;
        js.at("n_sq") = temp * js.at("n_queues_per_rtsm").template get<uint32_t>();
        printf("%s: set \"max\" n_sq to %u\n", __func__, js.at("n_sq").template get<uint32_t>());
    }
    nlohmann::json wq_js = js.at("n_wq");
    if ((wq_js.is_string()) && (wq_js.template get<std::string>().compare("max") == 0)) {
        uint32_t temp = max_rt_SMs;
        js.at("n_wq") = temp * js.at("n_queues_per_rtsm").template get<uint32_t>();
        printf("%s: set \"max\" n_wq to %u\n", __func__, js.at("n_wq").template get<uint32_t>());
    }

    try {
        gio_h.n_stage = js.at("n_stage");

        gio_h.n_sq = js.at("n_sq");
        gio_h.n_wq = js.at("n_wq");
        gio_h.sq_threshold = js.at("sq_threshold");
        gio_h.wq_threshold = js.at("wq_threshold");
        gio_h.n_queues_per_rtsm = js.at("n_queues_per_rtsm");
        force_assert(gio_h.n_sq > 0);
        force_assert(gio_h.n_stage > 0);
        force_assert(n_sq_entries > gio_h.sq_threshold * 2);
        force_assert(n_wq_entries > gio_h.wq_threshold * 2);

        prp_dma_num_pages_temp = js.at("prp_dma_num_pages");

        // Must be power of two
        // force_assert((aopt.n_sq_entries & (aopt.n_sq_entries - 1)) == 0);
        // force_assert((aopt.num_lib_cq_entries & (aopt.num_lib_cq_entries - 1)) == 0);

        force_assert(gio_h.n_sq == gio_h.n_wq);

    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        force_assert(false);
    }

    if (g_cuda_js.at("mode").template get<std::string>().compare("direct") == 0) {
        gio_h.total_num_lib_warps = (int)g_cuda_js.at("n_rt_blocks") *
                                    (int)g_cuda_js.at("n_rt_warps");
        gio_h.total_num_user_warps = 0;

        uint32_t n_warps_per_rtsm = g_cuda_js.at("n_rt_warps");
        uint32_t n_queues_per_rtsm = js.at("n_queues_per_rtsm");
        force_assert_printf(n_warps_per_rtsm >= n_queues_per_rtsm,
                            "%s: not enough warps to poll on all queues, %u %u\n", __func__, n_warps_per_rtsm, n_queues_per_rtsm);
    } else {
        force_assert_printf(false, "%s: Invalid mode.\n", __func__);
    }

    printf("%s: n_sq: %d, n_stage: %d, n_sq_entries: %d, n_stage_entries: %d\n",
           __func__, gio_h.n_sq, gio_h.n_stage, n_sq_entries, n_stage_entries);
    printf("%s: mode: %s, number of lib warps: %u, user warps: %u\n",
           __func__, g_cuda_js.at("mode").template get<std::string>().c_str(), gio_h.total_num_lib_warps, gio_h.total_num_user_warps);
}

void gio_w::print_stats(nlohmann::json &out_js) {
    class gio dg_temp_h;
    cudaErrChk(cudaMemcpy(&dg_temp_h, gio_dp, sizeof(dg_temp_h), cudaMemcpyDeviceToHost));
    dg_temp_h.print_stats(out_js.at("main_cuda_event_ms"), out_js);

    uint64_t runtime_n_ops = 0;
    for (uint32_t i = 0; i < dg_temp_h.n_sq; ++i) {
        struct lqueue_ptr<sq_ptr_t> temp;
        temp.value = 0;
        cudaErrChk(cudaMemcpy(&(temp), &(dg_temp_h.s_queue[i].lq.tail), sizeof(temp), cudaMemcpyDeviceToHost));
        runtime_n_ops += temp.value;
    }

    double runtime_iops = 0;
    float main_cuda_event_ms = out_js.at("main_cuda_event_ms");
    runtime_iops = (double)(runtime_n_ops * 1000) / (double)main_cuda_event_ms;
    printf("%s: runtime_n_ops: %lu, ms: %.3f, runtime_iops: %.3lf (using sq_tail)\n",
           __func__, runtime_n_ops, main_cuda_event_ms, runtime_iops);

    out_js.emplace("runtime_n_ops", runtime_n_ops);
    out_js.emplace("runtime_iops", runtime_iops);
}

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include <cuda/std/atomic>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>

#include <cuda.h>
#include <nvml.h>

#include "CMakeHeader.h"

#include "helper_headers/json.hpp"

#include "helper_headers/bench_config.cuh"
#include "helper_headers/color.h"
#include "helper_headers/fassert.cuh"
#include "helper_headers/gpuErrChk.h"
#include "helper_headers/gpuProp.cuh"
#include "helper_headers/convert.hpp"

#include "cudaProfiler.h"
#include <nvtx3/nvToolsExt.h>

#include "ds.cuh"
#include "addon.cuh"
#include "kernel.cuh"
#include "threads.cuh"

#include "work_ds.cuh"

#include "ctrl.h"
#include "nvme.h"

int main(int argc, char *argv[]) {
    printf("%s: Hello, %s, %s! ", __func__, C_PROJ_NAME, C_WORK_NAME);
    if (argc != 1) {
        fprintf(stderr, "%s: Error: do not provide arguments.\n", __func__);
        exit(EXIT_FAILURE);
    }
    delete_prefix_match(".", "result-" C_WORK_NAME);
    delete_prefix_match(".", "out-" C_WORK_NAME);

    std::string result_json_filename = "out-" C_WORK_NAME ".json";
    std::remove(result_json_filename.c_str());

    nlohmann::json options_js = convert_toml_to_json("options_in.toml");
    nlohmann::json out_js{};
    out_js.emplace("a:work_name", C_WORK_NAME);
    out_js.emplace("a:input_options_js", options_js);

    nlohmann::json cuda_js = options_js.at("g_cuda");
    nlohmann::json gio_js = options_js.at("g_io");

    cudaErrChk(cudaSetDevice(cuda_js.at("cudaDevice")));
    gpuProp(cuda_js.at("cudaDevice"), cuda_js.at("gpu_clock_mhz"));

    cuErrChk(cuInit(0));

    CUmoduleLoadingMode mode;
    cuErrChk(cuModuleGetLoadingMode(&mode));

    // Get GPU resource
    CUdevResource *orig_resource = new CUdevResource{};
    cuErrChk(cuDeviceGetDevResource(0, orig_resource, CU_DEV_RESOURCE_TYPE_SM));

    constexpr unsigned int nbGroups_max = 128;
    CUdevResource *split_resource = new CUdevResource[nbGroups_max]{};
    CUdevResource *remaining_resource = new CUdevResource{};
    unsigned int nbGroups = nbGroups_max;
    unsigned int useFlags = 0;
    unsigned int minCount = cuda_js.at("green_group_size");
    force_assert_printf(minCount >= 4, "invalid minimum count (group size).\n");
    cuErrChk(cuDevSmResourceSplitByCount(split_resource, &nbGroups, orig_resource, remaining_resource, useFlags, minCount));
    printf("%s: group size (minCount): %u, split actual nbGroups: %u\n", __func__, minCount, nbGroups);

    if (cuda_js.at("mode").template get<std::string>().compare("direct") == 0) {
        uint32_t max_rt_SMs = cuda_js.at("green_rt_n_groups").template get<uint32_t>() *
                              cuda_js.at("green_group_size").template get<uint32_t>();

        nlohmann::json n_work_sm_js = cuda_js.at("green_work_n_groups");
        if (n_work_sm_js.is_string() && n_work_sm_js.template get<std::string>().compare("max") == 0) {
            cuda_js.at("green_work_n_groups") = nbGroups - cuda_js.at("green_rt_n_groups").template get<uint32_t>();
        }
        uint32_t max_work_SMs = cuda_js.at("green_work_n_groups").template get<uint32_t>() *
                                cuda_js.at("green_group_size").template get<uint32_t>();

        nlohmann::json n_rt_warps_js = cuda_js.at("n_rt_warps");
        if ((n_rt_warps_js.is_string()) && (n_rt_warps_js.template get<std::string>().compare("max") == 0)) {
            int min_grid_size;
            int block_size;
            cudaErrChk(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, g_runtime_kernel));
            force_assert(block_size % 32 == 0);
            cuda_js.at("n_rt_warps") = block_size / 32;
            printf("%s: set num_lib_warps to %d\n", __func__, block_size / 32);
        }

        nlohmann::json n_rt_blocks_js = cuda_js.at("n_rt_blocks");
        if ((n_rt_blocks_js.is_string()) && (n_rt_blocks_js.template get<std::string>().compare("max") == 0)) {
            int nB;
            int blockSize = cuda_js.at("n_rt_warps").template get<int>() * 32;
            cudaErrChk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nB, g_runtime_kernel, blockSize, 0));
            cuda_js.at("n_rt_blocks") = max_rt_SMs * (uint32_t)nB;
            // cuda_js.at("n_rt_blocks") = max_rt_SMs * 5;
            printf("%s: set max num_lib_blocks to %u\n", __func__, max_rt_SMs * (uint32_t)nB);
        }
    }

    class gio_w my_gio{};
    my_gio.copy_options(cuda_js, gio_js);

    // Init nvme
    class g_nvme g_nvme{};
    g_nvme.init_nvme(cuda_js.at("cudaDevice"), options_js.at("nvme"));

    // Init gio
    my_gio.gio_w_init_h(g_nvme.ctrls);

    // Open work options
    std::string work_opt_filename = "work_in-" C_WORK_NAME ".toml";
    force_assert_printf(std::filesystem::exists(work_opt_filename), "filename: %s\n", work_opt_filename.c_str());
    nlohmann::json work_js = convert_toml_to_json(work_opt_filename);
    out_js.emplace("a:input_work_js", work_js);

    // Init work data
    struct work_w my_work{};
    my_work.work_w_init_h(options_js.at("nvme").at("num_controllers"), work_js.at("work"));

    // Init addon data
    class addon_w my_addon{};
    my_addon.addon_w_init_h(my_gio.get_total_n_lib_warps(), my_gio.get_total_n_user_warps(),
                            g_nvme.ctrls, work_js, my_work.bam_array_in_vec);

    struct thread_data *td = new struct thread_data(std::ref(out_js));
    td->cuda_js = &cuda_js;
    td->giop = &my_gio;
    td->addp = &my_addon;
    td->workp = &my_work;

    // Create green context
    int split_rt_n_groups = cuda_js.at("green_rt_n_groups");
    int split_work_n_groups = cuda_js.at("green_work_n_groups");
    CUdevResourceDesc *phDesc_split = new CUdevResourceDesc;
    CUdevResourceDesc *phDesc_user = new CUdevResourceDesc;
    cuErrChk(cuDevResourceGenerateDesc(phDesc_split, split_resource, split_rt_n_groups));
    cuErrChk(cuDevResourceGenerateDesc(phDesc_user, split_resource + split_rt_n_groups, split_work_n_groups));

    td->phCtx_lib = new CUgreenCtx;
    td->phCtx_user = new CUgreenCtx;
    int cu_dev = 0;
    cuErrChk(cuGreenCtxCreate(td->phCtx_lib, *phDesc_split, cu_dev, CU_GREEN_CTX_DEFAULT_STREAM));
    cuErrChk(cuGreenCtxCreate(td->phCtx_user, *phDesc_user, cu_dev, CU_GREEN_CTX_DEFAULT_STREAM));

    CUdevResource green_lib_resource;
    CUdevResource green_user_resource;
    cuErrChk(cuGreenCtxGetDevResource(*td->phCtx_lib, &green_lib_resource, CU_DEV_RESOURCE_TYPE_SM));
    cuErrChk(cuGreenCtxGetDevResource(*td->phCtx_user, &green_user_resource, CU_DEV_RESOURCE_TYPE_SM));
    printf("%s: actual sm.smCount (rt): %u, sm.smCount (work): %u\n",
           __func__, green_lib_resource.sm.smCount, green_user_resource.sm.smCount);
    force_assert(minCount * split_rt_n_groups == green_lib_resource.sm.smCount);
    if (cuda_js.at("green_work_n_groups") != 0) {
        force_assert(minCount * split_work_n_groups == green_user_resource.sm.smCount);
    }

    std::vector<std::thread> threads;
    threads.emplace_back(thread_init, td);
    threads.back().join();
    threads.pop_back();

    // Launch kernels
    if (cuda_js.at("mode").template get<std::string>().compare("direct") == 0) {
        // Launch separate threads (and kernels)
        threads.emplace_back(thread_runtime, td);

        // threads.emplace_back(thread_app, td);
        std::thread thread_application(thread_app, td);
        thread_application.join();

        // Shutdown runtime
        std::unique_lock lk(td->mut);
        td->runtime_shutdown = true;
        td->cv.notify_all();
        lk.unlock();
    } else {
        force_assert(false);
    }

    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }

    cudaErrChk(cudaDeviceSynchronize());
    cuErrChk(cuGreenCtxDestroy(*td->phCtx_user));
    cuErrChk(cuGreenCtxDestroy(*td->phCtx_lib));
    delete td->phCtx_user;
    delete td->phCtx_lib;

    delete phDesc_user;
    delete phDesc_split;
    delete remaining_resource;
    delete[] split_resource;
    delete orig_resource;

    delete td;

    my_addon.udata_out(cuda_js.at("gpu_clock_mhz"), out_js);
    my_addon.addon_w_fini_h();

    my_work.work_print_stats(cuda_js.at("gpu_clock_mhz"), out_js);
    my_work.work_w_fini_h();

    my_gio.print_stats(out_js);
    my_gio.gio_w_fini_h();

    g_nvme.fini_nvme();

    std::ofstream of;
    of.open(result_json_filename.c_str());
    of << out_js.dump(4) << std::endl;
    of.close();

    printf("%s: Bye!\n", __func__);
    return 0;
}

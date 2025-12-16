#pragma once

#include <vector>
#include <cstdio>
#include <ctime>

#include <cuda.h>

void launch_sm_runtime_kernel(class thread_data *td, const uint32_t n_rt_blocks, const uint32_t n_rt_warps,
                              std::vector<CUstream> &streams) {
    printf("%s: launching runtime with %u blocks * %u threads...\n", __func__, n_rt_blocks, n_rt_warps * 32);
    void *lib_kernelArgs[2] = {&(td->giop->gio_dp), &(td->addp->addon_dp)};
    cudaErrChk(cudaLaunchKernel((void *)g_runtime_kernel, n_rt_blocks, n_rt_warps * 32,
                                lib_kernelArgs, 0, streams[0]));
}

void launch_sm_kernel(class thread_data *td, class thread_user_data &tu) {
    nlohmann::json out_js;

    void *g_app_kernel_init_args[1] = {&(td->giop->gio_dp)};
    cudaErrChk(cudaLaunchKernel(g_app_kernel_init, 1024, 128, g_app_kernel_init_args, 0, *tu.phStream));
    cudaErrChk(cudaStreamSynchronize(*tu.phStream));

    work_launcher(out_js, *td->cuda_js, *td->giop, *td->workp, tu);

    std::unique_lock lk(td->mut);
    td->out_js->merge_patch(out_js);
    lk.unlock();
}

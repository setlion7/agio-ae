#pragma once

#include <cuda.h>
#include <mutex>
#include <vector>

#include "ds.cuh"
#include "types_h.cuh"

#include "launcher.cuh"

void thread_init(struct thread_data *td) {
    cudaFuncAttributes attr;
    cudaErrChk(cudaFuncGetAttributes(&attr, g_runtime_kernel));
    cudaErrChk(cudaFuncGetAttributes(&attr, g_app_kernel_init));
    cudaErrChk(cudaFuncGetAttributes(&attr, gio_rt_init));
    cudaErrChk(cudaFuncGetAttributes(&attr, gio_rt_fini));
    cudaErrChk(cudaFuncGetAttributes(&attr, gio_rt_shutdown));
    cudaErrChk(cudaFuncGetAttributes(&attr, app_kernel_init));
    cudaErrChk(cudaFuncGetAttributes(&attr, app_kernel_fini));
    cudaErrChk(cudaFuncGetAttributes(&attr, addon_kernel_init));
    cudaErrChk(cudaFuncGetAttributes(&attr, addon_kernel_fini));

    work_kernel_register();
}

void thread_runtime(struct thread_data *td) {
    printf("%s: thread start.\n", __func__);

    CUcontext *pContext = new CUcontext;
    cuErrChk(cuCtxFromGreenCtx(pContext, *(td->phCtx_lib)));
    cuErrChk(cuCtxSetCurrent(*pContext));

    std::vector<CUstream> streams(2);
    for (size_t i = 0; i < streams.size(); ++i)
        cuErrChk(cuGreenCtxStreamCreate(&streams[i], *(td->phCtx_lib), CU_STREAM_NON_BLOCKING, 0));

    nlohmann::json &cuda_js = *td->cuda_js;
    const uint32_t n_rt_blocks = cuda_js.at("n_rt_blocks");
    const uint32_t n_rt_warps = cuda_js.at("n_rt_warps");

    class gio *gio_dp = td->giop->gio_dp;

    void *addon_kernel_args[2] = {&gio_dp, &(td->addp->addon_dp)};
    cudaErrChk(cudaLaunchKernel((void *)addon_kernel_init, n_rt_blocks, n_rt_warps * 32, addon_kernel_args, 0, streams[0]));
    cudaErrChk(cudaStreamSynchronize(streams[0]));

    void *gio_rt_args[1] = {&(gio_dp)};
    cudaErrChk(cudaLaunchKernel((void *)gio_rt_init, n_rt_blocks, n_rt_warps * 32, gio_rt_args, 0, streams[0]));
    cudaErrChk(cudaStreamSynchronize(streams[0]));

    if (cuda_js.at("mode").template get<std::string>().compare("direct") == 0) {
        launch_sm_runtime_kernel(td, n_rt_blocks, n_rt_warps, streams);
    } else {
        force_assert(false);
    }

    // Wait
    std::unique_lock lk(td->mut);
    td->runtime_launched = true;
    td->cv.notify_all();
    td->cv.wait(lk, [&td] { return td->runtime_shutdown; });
    lk.unlock();

    cudaErrChk(cudaLaunchKernel(gio_rt_shutdown, 1, 128, gio_rt_args, 0, streams[1]));
    cudaErrChk(cudaStreamSynchronize(streams[1]));

    cudaErrChk(cudaLaunchKernel((void *)gio_rt_fini, 1, 1, gio_rt_args, 0, streams[0]));
    cudaErrChk(cudaStreamSynchronize(streams[0]));

    cudaErrChk(cudaLaunchKernel((void *)addon_kernel_fini, n_rt_blocks, n_rt_warps * 32, addon_kernel_args, 0, streams[0]));
    cudaErrChk(cudaStreamSynchronize(streams[0]));

    for (size_t i = 0; i < streams.size(); ++i)
        cudaErrChk(cudaStreamSynchronize(streams[i]));
    for (size_t i = 0; i < streams.size(); ++i)
        cuErrChk(cuStreamDestroy(streams[i]));

    CUcontext temp_ctx;
    cuCtxGetCurrent(&temp_ctx);
    force_assert(temp_ctx == *pContext);

    delete pContext;

    printf("%s: thread end.\n", __func__);
}

void thread_app(class thread_data *td) {
    printf("%s: thread start.\n", __func__);

    nlohmann::json &cuda_js = *td->cuda_js;
    class gio *gio_dp = td->giop->gio_dp;

    // Wait for runtime
    std::unique_lock lk(td->mut);
    td->cv.wait(lk, [&td] { return td->runtime_launched; });
    lk.unlock();

    class thread_user_data tu;

    cuErrChk(cuGreenCtxStreamCreate(tu.phStream, *(td->phCtx_user), CU_STREAM_NON_BLOCKING, 0));
    cuErrChk(cuCtxFromGreenCtx(tu.pContext, *(td->phCtx_user)));
    cuErrChk(cuCtxSetCurrent(*tu.pContext));

    // Check runtime status
    uint64_t num_addr = (uint64_t)gio_dp + offsetof(class gio, rt_running);
    cuErrChk(cuStreamWaitValue32(*tu.phStream, num_addr, 1, CU_STREAM_WAIT_VALUE_EQ));

    void *app_kernel_args[3] = {&(gio_dp), &(td->addp->addon_dp), &(td->workp->wd_dp)};
    cudaErrChk(cudaLaunchKernel((void *)app_kernel_init, 1, 1, app_kernel_args, 0, *tu.phStream));
    cudaErrChk(cudaStreamSynchronize(*tu.phStream));

    // Launch app kernels
    if (cuda_js.at("mode").template get<std::string>().compare("direct") == 0) {
        launch_sm_kernel(td, tu);
    } else {
        force_assert(false);
    }

    cudaErrChk(cudaLaunchKernel((void *)app_kernel_fini, 1, 1, app_kernel_args, 0, *tu.phStream));
    cudaErrChk(cudaStreamSynchronize(*tu.phStream));

    // Time
    float main_cuda_event_ms = 0;
    cudaErrChk(cudaEventElapsedTime(&main_cuda_event_ms, tu.event_start, tu.event_end));
    uint64_t main_ts_duration_ns = (tu.ts_end.tv_sec - tu.ts_start.tv_sec) * 1000000000UL +
                                   (tu.ts_end.tv_nsec - tu.ts_start.tv_nsec);
    printf(CGREEN "%s: cudaEventElapsedTime() duration: %.3f ms\n" CRESET, __func__, main_cuda_event_ms);
    printf(CGREEN "%s: clock_gettime() duration: %.3lf ms (%.3lf us)\n" CRESET,
           __func__, (double)main_ts_duration_ns / 1000.0 / 1000.0, (double)main_ts_duration_ns / 1000.0);

    lk.lock();
    nlohmann::json &out_js = *td->out_js;
    out_js["main_cuda_event_ms"] = main_cuda_event_ms;
    out_js["main_ts_duration_ns"] = main_ts_duration_ns;
    lk.unlock();

    cuErrChk(cuStreamDestroy(*tu.phStream));

    CUcontext temp_ctx;
    cuCtxGetCurrent(&temp_ctx);
    force_assert(temp_ctx == *tu.pContext);

    printf("%s: thread end.\n", __func__);
}

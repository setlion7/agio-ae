#pragma once

#include <mutex>
#include <condition_variable>
#include "cuda.h"

class thread_data {
    public:
    std::mutex mut;
    std::condition_variable cv;

    bool runtime_launched;
    bool runtime_shutdown;

    // CUDA Green Context
    CUgreenCtx *phCtx_lib;
    CUgreenCtx *phCtx_user;

    // Launch options
    nlohmann::json *cuda_js;

    class gio_w *giop;
    class addon_w *addp;
    class work_w *workp;

    // json
    nlohmann::json *out_js;

    thread_data(nlohmann::json &out_js_in)
        : mut(), cv(), runtime_launched(false), runtime_shutdown(false), out_js(&out_js_in) {};
    ~thread_data() {};
};

class thread_user_data {
    public:
    struct timespec ts_start, ts_end;
    cudaEvent_t event_start, event_end;

    CUstream *phStream;
    CUcontext *pContext;

    thread_user_data() {
        cudaErrChk(cudaEventCreate(&event_start));
        cudaErrChk(cudaEventCreate(&event_end));

        phStream = new CUstream;
        pContext = new CUcontext;
    }
    ~thread_user_data() {
        delete pContext;
        delete phStream;

        cudaErrChk(cudaEventDestroy(event_start));
        cudaErrChk(cudaEventDestroy(event_end));
    }
};

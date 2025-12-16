#pragma once

#if G_BENCH_CPU_ENABLE == 1

#define G_BENCH_CPU_APP_START() \
    float acc_time_ms = 0.0;    \
    float level_time_ms = 0.0;

#define G_BENCH_CPU_LEVEL_START() \
    auto startt = std::chrono::system_clock::now();

#define G_BENCH_CPU_LEVEL_PRINT 0

#if G_BENCH_CPU_LEVEL_PRINT == 1
#define G_BENCH_CPU_LEVEL_PRINTF(level)                           \
    printf("%s: level: %u, time(ms): %.3f, acc time(ms): %.3f\n", \
           __func__, level, level_time_ms, acc_time_ms);
#elif G_BENCH_CPU_LEVEL_PRINT == 0
#define G_BENCH_CPU_LEVEL_PRINTF(level)
#endif

#define G_BENCH_CPU_LEVEL_END(out_js, level)                                               \
    {                                                                                      \
        auto endt = std::chrono::system_clock::now();                                      \
        level_time_ms = std::chrono::duration<float, std::milli>(endt - startt).count();   \
        acc_time_ms += level_time_ms;                                                      \
        out_js["g_bench_cpu_level_time_ms"].emplace(std::to_string(level), level_time_ms); \
        out_js["g_bench_cpu_level_acc_ms"].emplace(std::to_string(level), acc_time_ms);    \
        G_BENCH_CPU_LEVEL_PRINTF(level);                                                   \
    }

#define G_BENCH_CPU_APP_END(out_js, level) \
    out_js.emplace("g_bench_cpu_n_level", level);

#elif G_BENCH_CPU_ENABLE == 0

#define G_BENCH_CPU_APP_START()
#define G_BENCH_CPU_LEVEL_START()
#define G_BENCH_CPU_LEVEL_END(out_js, level)
#define G_BENCH_CPU_APP_END()

#endif

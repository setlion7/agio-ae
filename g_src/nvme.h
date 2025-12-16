#pragma once

#include "helper_headers/json.hpp"

#include "ctrl.h"

class g_nvme {
  public:
    std::vector<Controller *> ctrls;

    // From json
    uint32_t num_controllers;
    std::vector<std::string> nvme_path;
    uint32_t nvme_namespace;
    size_t queue_depth;
    size_t num_queue_pairs;

    void init_nvme(int cudaDevice, const nlohmann::json js_nvme);
    void fini_nvme();
};

void g_nvme::init_nvme(int cudaDevice, const nlohmann::json js_nvme) {
    try {
        num_controllers = js_nvme.at("num_controllers");
        nvme_path = js_nvme.at("nvme_path");
        nvme_namespace = js_nvme.at("namespace");
        queue_depth = js_nvme.at("queue_depth");
        num_queue_pairs = js_nvme.at("num_queue_pairs");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        force_assert(false);
    }
    // force_assert(num_controllers == nvme_path.size());
    // opt->num_controllers = num_controllers;

    ctrls.resize(num_controllers);
    try {
        for (size_t i = 0; i < num_controllers; i++) {
            ctrls[i] = new Controller(nvme_path[i].c_str(), nvme_namespace,
                                      cudaDevice, queue_depth, num_queue_pairs);
            // printf("Controller %s created.\n", myopt.iopt.nvme_path[i].c_str());
            // printf("Max data transfer size: %lu bytes\n", ctrls[i]->info.max_data_size);
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "%s: Failed to create controller. Reason: %s\n", __func__, e.what());
        force_assert(false);
    }
    printf("%s: num_controllers: %u, queue_depth: %lu\n", __func__, num_controllers, queue_depth);
}

void g_nvme::fini_nvme() {
    for (size_t i = 0; i < num_controllers; i++)
        delete ctrls[i];
    // printf("%s: done.\n", __func__);
}

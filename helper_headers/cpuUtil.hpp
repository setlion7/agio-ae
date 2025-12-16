#pragma once

#include <filesystem>
void delete_prefix_match(std::filesystem::path dir_path, std::string prefix) {
    for (const auto &entry : std::filesystem::directory_iterator(dir_path)) {
        const auto &filepath = entry.path();
        const std::string filename = filepath.filename().string();
        if (filename.rfind(prefix, 0) == 0) {
            printf("%s: removing filename: %s\n", __func__, filename.c_str());
            std::filesystem::remove(filepath);
        }
    }
}

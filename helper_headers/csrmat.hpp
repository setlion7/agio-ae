#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

#include "json.hpp"

template <typename vertex_t, typename value_t>
class Csrmat {
    public:
    std::string mat_filename;
    uint64_t num_rows;
    uint64_t num_cols;
    uint64_t num_non_zeros;
    uint64_t edge_list_len;
    vertex_t *vertex_list;
    vertex_t *edge_list;

    bool symmetric;
    bool pattern;
    value_t *value_list;

    Csrmat(std::string filename) {
        mat_filename = filename;
        num_rows = 0;
        num_cols = 0;
        num_non_zeros = 0;
        edge_list_len = 0;
        vertex_list = nullptr;
        edge_list = nullptr;
        value_list = nullptr;

        symmetric = false;
        pattern = false;
    }
    ~Csrmat() {
        if (vertex_list)
            free(vertex_list);
        if (edge_list)
            free(edge_list);
        if (value_list)
            free(value_list);
    }

    void read_mat() {
        check_meta();
        read_vertex();
        // read_edge();
        // read_value();
    }

    void check_meta() {
        std::ifstream input_file(mat_filename + ".json");
        if (!input_file.is_open()) {
            fprintf(stderr, "%s: file open failed.\n", __func__);
            exit(EXIT_FAILURE);
        }

        nlohmann::json js;
        js = nlohmann::json::parse(input_file);
        input_file.close();

        std::string mat_value;
        std::string mat_value_type;

        try {
            mat_value = js.at("value");
            mat_value_type = js.at("value_type");
            num_rows = js.at("num_rows");
            num_cols = js.at("num_cols");
            num_non_zeros = js.at("num_non_zeros");
            edge_list_len = js.at("edge_list_len");
            symmetric = js.at("symmetric");
        } catch (const std::exception &e) {
            std::cerr << e.what() << ", file: " << __FILE__ << ", line: " << __LINE__ << std::endl;
            exit(EXIT_FAILURE);
        }

        if (symmetric) {
            if (num_rows != num_cols) {
                fprintf(stderr, "%s: invalid meta file.\n", __func__);
                exit(EXIT_FAILURE);
            }
            if (num_non_zeros * 2 != edge_list_len) {
                fprintf(stderr, "%s: invalid NNZ or edge_list_len\n", __func__);
                exit(EXIT_FAILURE);
            }
        } else {
            if (num_non_zeros != edge_list_len) {
                fprintf(stderr, "%s: invalid NNZ or edge_list_len\n", __func__);
                exit(EXIT_FAILURE);
            }
        }

        if (mat_value.compare("pattern") == 0) {
            pattern = true;
        } else if (typeid(value_t) == typeid(void)) {
            // Not using values
        } else if (mat_value.compare("integer") == 0) {
            if (mat_value_type.compare("uint32_t") == 0) {
                if ((typeid(value_t) == typeid(int32_t)) || (typeid(value_t) == typeid(uint32_t))) {
                    printf("%s: info: value_t is 32bit integer...\n", __func__);
                } else
                    goto meta_error;
                pattern = false;
            } else if (mat_value_type.compare("uint64_t") == 0) {
                if ((typeid(value_t) == typeid(int64_t)) || (typeid(value_t) == typeid(uint64_t))) {
                    // OK
                } else
                    goto meta_error;
                pattern = false;
            }
        } else if (mat_value.compare("real") == 0) {
            if (mat_value_type.compare("float") == 0) {
                if (typeid(value_t) == typeid(float)) {
                    printf("%s: info: value_t is 32bit float...\n", __func__);
                } else
                    goto meta_error;
                pattern = false;
            } else if (mat_value_type.compare("double") == 0) {
                if (typeid(value_t) == typeid(double)) {
                    // OK
                } else
                    goto meta_error;
                pattern = false;
            }
        } else {
        meta_error:
            fprintf(stderr, "%s: invalid meta file or type. mat_value: %s, mat_value_type: %s, but value_t: %s\n",
                    __func__, mat_value.c_str(), mat_value_type.c_str(), typeid(value_t).name());
            exit(EXIT_FAILURE);
        }
        // printf("%s: pattern matrix? %s\n", __func__, pattern? "true":"false");
    }

    void read_vertex() {
        // Read vertex file
        size_t ret;
        std::string vertex_filename = mat_filename + ".vertex";
        FILE *vertex_file = fopen(vertex_filename.c_str(), "r");
        vertex_list = (vertex_t *)malloc(sizeof(vertex_t) * (num_rows + 1));
        if (!vertex_list) {
            perror("malloc() error");
            exit(EXIT_FAILURE);
        }
        memset(vertex_list, 0, sizeof(vertex_t) * (num_rows + 1));
        ret = fread(vertex_list, sizeof(vertex_t), num_rows + 1, vertex_file);
        if (ret != num_rows + 1) {
            fprintf(stderr, "Vertex file read error\n");
            exit(EXIT_FAILURE);
        }
        fclose(vertex_file);
        printf("%s: done.\n", __func__);
    }

    void read_edge() {
        // Read edge file
        size_t ret;
        std::string edge_filename = mat_filename + ".edge";
        FILE *edge_file = fopen(edge_filename.c_str(), "r");
        edge_list = (vertex_t *)malloc(sizeof(vertex_t) * edge_list_len);
        if (!edge_list) {
            perror("malloc() error");
            exit(EXIT_FAILURE);
        }
        ret = fread(edge_list, sizeof(vertex_t), edge_list_len, edge_file);
        if (ret != edge_list_len) {
            fprintf(stderr, "Edge file read error, ret: %lu\n", ret);
            exit(EXIT_FAILURE);
        }
        fclose(edge_file);
        printf("%s: done.\n", __func__);
    }

    void read_value() {
        if (pattern) {
            fprintf(stderr, "This is a pattern matrix.\n");
            exit(EXIT_FAILURE);
        } else if (typeid(value_t) == typeid(void)) {
            fprintf(stderr, "Invalid value type.\n");
            exit(EXIT_FAILURE);
        }

        // Read value file
        size_t ret;
        std::string value_filename = mat_filename + ".value";
        FILE *value_file = fopen(value_filename.c_str(), "r");
        value_list = (value_t *)malloc(sizeof(value_t) * edge_list_len);
        if (!value_list) {
            perror("malloc() error");
            exit(EXIT_FAILURE);
        }
        ret = fread(value_list, sizeof(value_t), edge_list_len, value_file);
        if (ret != edge_list_len) {
            fprintf(stderr, "Value file read error\n");
            exit(EXIT_FAILURE);
        }
        fclose(value_file);
        printf("%s: done.\n", __func__);
    }
};

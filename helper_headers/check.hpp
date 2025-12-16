#pragma once

#include <cstdio>
#include <cstdlib>
#include <typeinfo>

#define g_typecheck(typeid_in, str_in)                                    \
    bool type_error = false;                                              \
    if (typeid_in == typeid(double) && ("double" != str_in)) {            \
        type_error = true;                                                \
    } else if (typeid_in == typeid(float) && ("float" != str_in)) {       \
        type_error = true;                                                \
    } else if (typeid_in == typeid(uint64_t) && ("uint64_t" != str_in)) { \
        type_error = true;                                                \
    } else if (typeid_in == typeid(uint32_t) && ("uint32_t" != str_in)) { \
        type_error = true;                                                \
    }                                                                     \
    if (type_error) {                                                     \
        fprintf(stderr, "%s:%s: Invalid type.\n", __FILE__, __func__);    \
        exit(EXIT_FAILURE);                                               \
    }

#pragma once

#include <string>
#include <cstdio>
#include <iomanip>
#include <iostream>

#include "json.hpp"

#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1
#include "toml.hpp"

nlohmann::json convert_toml_to_json(std::string toml_filename)
{
    toml::table data = toml::parse_file(toml_filename);
    std::stringstream ss;
    ss << toml::json_formatter{ data } << std::endl;

    nlohmann::json in_json;
    try
    {
        in_json = nlohmann::json::parse(ss.str());
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string converted_filename = toml_filename + "_converted.json";
    std::remove(converted_filename.c_str());
    std::ofstream of;
    of.open(converted_filename);
    of << in_json.dump(4) << std::endl;
    of.close();

    return in_json;
}

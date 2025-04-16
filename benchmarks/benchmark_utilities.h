#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "misc/datatypes.h"
#include "misc/dataloader.h"

std::string get_date_string() {
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(now, "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

typedef struct infer_config {
    int target_crt_base_size;
    vector<float> relu_accs;
    dataset_tensor dataset;
    std::string model_name;
    std::string model_file;
    QuantizationMethod quantization_method;
    int q_parameter; // this is l for ScaleQuant, s for ScaleQuantPlus, ignored for SimpleQuant
    bool optimize_bases;
} infer_config_t;

void create_dir(std::string path) {
    try {
        std::filesystem::create_directory(path);
    } catch (std::filesystem::filesystem_error& e) {
        std::cout << e.what() << std::endl;
    }
}
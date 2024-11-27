#ifndef DATALOADER_H
#define DATALOADER_H

#ifndef SGX
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/matrix.h>
#include <dlib/pixel.h>
#endif

#include <cstdlib>
#include <vector>

#include "circuit/scalar_tensor.h"
#include "misc/datatypes.h"

using std::vector;

struct dataset_tensor {
    vector<ScalarTensor<wandb_t>> training_images;
    vector<unsigned long> training_labels;
    vector<ScalarTensor<wandb_t>> testing_images;
    vector<unsigned long> testing_labels;
};

struct dataset_tensor_q {
    vector<ScalarTensor<q_val_t>> training_images;
    vector<unsigned long> training_labels;
    vector<ScalarTensor<q_val_t>> testing_images;
    vector<unsigned long> testing_labels;
};

#ifndef SGX
struct mnist_dataset {
    vector<dlib::matrix<unsigned char>> training_images;
    vector<unsigned long> training_labels;
    vector<dlib::matrix<unsigned char>> testing_images;
    vector<unsigned long> testing_labels;
};

struct cifar10_dataset {
    vector<dlib::matrix<dlib::rgb_pixel>> training_images;
    vector<unsigned long> training_labels;
    vector<dlib::matrix<dlib::rgb_pixel>> testing_images;
    vector<unsigned long> testing_labels;
};

dataset_tensor mnist(std::string path) {
    mnist_dataset data;
    dlib::load_mnist_dataset(path, data.training_images, data.training_labels,
                             data.testing_images, data.testing_labels);

    dataset_tensor tensor_data;
    // input_width x input_height x input_channel
    dim_t image_dims{28, 28, 1};
    for (size_t i = 0; i < data.training_images.size(); ++i) {
        auto smart_data_ptr = data.training_images.at(i).steal_memory();
        auto data_ptr = smart_data_ptr.get();
        auto tensor = ScalarTensor<wandb_t>::create_with_cast<unsigned char>(
            data_ptr, image_dims);
        tensor /= 255.0;
        tensor_data.training_images.push_back(tensor);
    }
    tensor_data.training_labels = data.training_labels;

    for (size_t i = 0; i < data.testing_images.size(); ++i) {
        auto smart_data_ptr = data.testing_images.at(i).steal_memory();
        auto data_ptr = smart_data_ptr.get();
        auto tensor = ScalarTensor<wandb_t>::create_with_cast<unsigned char>(
            data_ptr, image_dims);
        tensor /= 255.0;
        tensor_data.testing_images.push_back(tensor);
    }
    tensor_data.testing_labels = data.testing_labels;

    return tensor_data;
}

dataset_tensor cifar10(std::string path) {
    cifar10_dataset data;
    dlib::load_cifar_10_dataset(path, data.training_images,
                                data.training_labels, data.testing_images,
                                data.testing_labels);

    dataset_tensor tensor_data;
    // input_width x input_height x input_channel
    dim_t image_dims{32, 32, 3};
    for (size_t i = 0; i < data.training_images.size(); ++i) {
        ScalarTensor<wandb_t> tensor(image_dims);
        for (size_t j = 0; j < 32; ++j) {
            for (size_t k = 0; k < 32; ++k) {
                tensor.set({j, k, 0},
                           data.training_images.at(i)(k, j).red / 255.0);
                tensor.set({j, k, 1},
                           data.training_images.at(i)(k, j).green / 255.0);
                tensor.set({j, k, 2},
                           data.training_images.at(i)(k, j).blue / 255.0);
            }
        }
        tensor_data.training_images.push_back(tensor);
    }
    tensor_data.training_labels = data.training_labels;

    for (size_t i = 0; i < data.testing_images.size(); ++i) {
        ScalarTensor<wandb_t> tensor(image_dims);
        for (size_t j = 0; j < 32; ++j) {
            for (size_t k = 0; k < 32; ++k) {
                tensor.set({j, k, 0},
                           data.testing_images.at(i)(k, j).red / 255.0);
                tensor.set({j, k, 1},
                           data.testing_images.at(i)(k, j).green / 255.0);
                tensor.set({j, k, 2},
                           data.testing_images.at(i)(k, j).blue / 255.0);
            }
        }
        tensor_data.testing_images.push_back(tensor);
    }
    tensor_data.testing_labels = data.testing_labels;

    return tensor_data;
}

dataset_tensor_q quantize(dataset_tensor* data, wandb_t q_const) {
    dataset_tensor_q quantized_data;
    for (auto input : data->training_images) {
        auto tensor = ScalarTensor<q_val_t>::quantize(
            input, QuantizationMethod::SimpleQuant, q_const);
        quantized_data.training_images.push_back(tensor);
    }
    quantized_data.training_labels = data->training_labels;

    for (auto input : data->testing_images) {
        auto tensor = ScalarTensor<q_val_t>::quantize(
            input, QuantizationMethod::SimpleQuant, q_const);
        quantized_data.testing_images.push_back(tensor);
    }
    quantized_data.testing_labels = data->testing_labels;

    return quantized_data;
}

void normalize(dataset_tensor* dataset, vector<wandb_t> mean,
               vector<wandb_t> std) {
    dim_t data_dims = dataset->training_images.at(0).get_dims();
    assert(mean.size() == data_dims.back() && "mean size mismatch");
    assert(std.size() == data_dims.back() && "std size mismatch");

#pragma omp parallel for
    for (size_t i = 0; i < dataset->training_images.size(); ++i) {
        for (size_t j = 0; j < data_dims[0]; ++j) {
            for (size_t k = 0; k < data_dims[1]; ++k) {
                for (size_t l = 0; l < data_dims[2]; ++l) {
                    wandb_t val = dataset->training_images.at(i).get({j, k, l});
                    val = (val - mean.at(l)) / std.at(l);
                    dataset->training_images.at(i).set({j, k, l}, val);
                }
            }
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < dataset->testing_images.size(); ++i) {
        for (size_t j = 0; j < data_dims[0]; ++j) {
            for (size_t k = 0; k < data_dims[1]; ++k) {
                for (size_t l = 0; l < data_dims[2]; ++l) {
                    wandb_t val = dataset->testing_images.at(i).get({j, k, l});
                    val = (val - mean.at(l)) / std.at(l);
                    dataset->testing_images.at(i).set({j, k, l}, val);
                }
            }
        }
    }
}
#endif

vector<ScalarTensor<q_val_t>> quantize(vector<ScalarTensor<wandb_t>>& data,
                                       QuantizationMethod q_method,
                                       wandb_t q_const) {
    vector<ScalarTensor<q_val_t>> quantized_data;
    for (auto input : data) {
        auto tensor = ScalarTensor<q_val_t>::quantize(input, q_method, q_const);
        quantized_data.push_back(tensor);
    }

    return quantized_data;
}

vector<ScalarTensor<q_val_t>> quantize(vector<ScalarTensor<wandb_t>>& data,
                                       QuantizationMethod q_method, int l) {
    vector<ScalarTensor<q_val_t>> quantized_data;
    for (auto input : data) {
        auto tensor = ScalarTensor<q_val_t>::quantize(input, q_method, l);
        quantized_data.push_back(tensor);
    }

    return quantized_data;
}

#endif
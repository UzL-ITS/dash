#ifndef ONNX_MODELLOADER_H
#define ONNX_MODELLOADER_H

#ifndef SGX
#include <sys/stat.h>
#endif

#include <fstream>

#include "circuit/circuit.h"
#include "circuit/layer/conv2d.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/flatten.h"
#include "circuit/layer/max_pool2d.h"
#include "circuit/layer/relu.h"
#include "circuit/layer/rescale.h"
#include "circuit/layer/sign.h"
#include "circuit/scalar_tensor.h"
#include "misc/datatypes.h"
#include "onnx.proto3.pb.h"

using std::vector;

#ifndef SGX
// https://stackoverflow.com/a/51301928/8538713
inline bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

// Modified from:
// https://stackoverflow.com/questions/67301475/parse-an-onnx-model-using-c-extract-layers-input-and-output-shape-from-an-on
onnx::ModelProto parse_model_file(std::string path) {
    if (!file_exists(path)) {
        std::cerr << "File does not exist: " << path << std::endl;
        exit(1);
    }

    std::ifstream input(
        path,
        std::ios::ate | std::ios::binary);  // open file and move current
                                            // position in file to the end

    std::streamsize size = input.tellg();  // get current position in file
    input.seekg(0, std::ios::beg);         // move to start of file

    vector<char> buffer(size);
    input.read(buffer.data(), size);  // read raw data

    onnx::ModelProto model;
    model.ParseFromArray(buffer.data(), size);  // parse protobuf

    return model;
}

// used in ocall
raw_data_t read_file(const char* path) {
    if (!file_exists(path)) {
        std::cerr << "File does not exist: " << path << std::endl;
        exit(1);
    }

    std::ifstream input(
        path,
        std::ios::ate | std::ios::binary);  // open file and move current
                                            // position in file to the end

    size_t size = input.tellg();    // get current position in file
    input.seekg(0, std::ios::beg);  // move to start of file

    char* buffer = (char*)malloc(size);

    input.read(buffer, size);  // read raw data

    return {buffer, size};
}
#else
onnx::ModelProto parse_model_file(std::string path) {
    raw_data_t raw_data;
    ocall_read_file(&raw_data, path.c_str());
    onnx::ModelProto model;
    model.ParseFromArray(raw_data.data, raw_data.size);  // parse protobuf
    ocall_free(raw_data.data);

    return model;
}
#endif

// From https://github.com/onnx/onnx/blob/main/onnx/common/platform_helpers.h
// Determine if the processor is little endian or not
inline bool is_processor_little_endian() {
    int num = 1;
    if (*(char*)&num == 1) return true;
    return false;
}

// Modified from:
// https://github.com/onnx/onnx/blob/main/onnx/defs/tensor_util.cc
const vector<float> parse_raw_data(std::string raw_data) {
    vector<float> res;
    // okay to remove const qualifier as we have already made a copy
    char* bytes = const_cast<char*>(raw_data.c_str());
    // onnx is little endian serialized always-tweak byte order if needed
    if (!is_processor_little_endian()) {
        const size_t element_size = sizeof(float);
        const size_t num_elements = raw_data.size() / element_size;
        for (size_t i = 0; i < num_elements; ++i) {
            char* start_byte = bytes + i * element_size;
            char* end_byte = start_byte + element_size - 1;
            /* keep swapping */
            for (size_t count = 0; count < element_size / 2; ++count) {
                char temp = *start_byte;
                *start_byte = *end_byte;
                *end_byte = temp;
                ++start_byte;
                --end_byte;
            }
        }
    }
    // raw_data.c_str()/bytes is a byte array and may not be properly
    // aligned for the underlying type
    // We need to copy the raw_data.c_str()/bytes as byte instead of
    // copying as the underlying type, otherwise we may hit memory
    // misalignment issues on certain platforms, such as arm32-v7a
    const size_t raw_data_size = raw_data.size();
    res.resize(raw_data_size / sizeof(float));
    memcpy(reinterpret_cast<char*>(res.data()), bytes, raw_data_size);
    return res;
}

int count_wandb_tensors(onnx::ModelProto model) {
    int count = 0;
    for (auto node : model.graph().node()) {
        if (node.op_type().compare("Gemm") == 0) {
            count += 2;
        } else if (node.op_type().compare("Conv") == 0) {
            count += 2;
        }
    }
    return count;
}

int get_nr_channel_before_flatten(vector<Layer*> layer) {
    int last_conv2d_channel = 0;
    for (auto l : layer) {
        if (l->get_type() == Layer::LayerType::conv2d) {
            last_conv2d_channel = ((Conv2d*)l)->get_filter();
        }
        if (l->get_type() == Layer::LayerType::flatten) {
            return last_conv2d_channel;
        }
    }
    return 0;
}

Circuit* create_circuit_from_onnx_model(onnx::ModelProto model,
                                        QuantizationMethod q_method,
                                        wandb_t q_const) {
    // get framwork name
    auto framework = model.producer_name();

    vector<Layer*> layer;
    auto graph = model.graph();
    int wandb_tensor_cnt = 0;
    auto wandb = graph.initializer();
    if (framework.compare("tf2onnx") == 0) {
        wandb_tensor_cnt = count_wandb_tensors(model) - 1;
    }

    // get model input dimensions
    auto d = model.graph().input(0).type().tensor_type().shape().dim();
    size_t channel;
    size_t input_height;
    size_t input_width;

    if (framework.compare("tf2onnx") == 0 && d.size() != 3 && d.size() != 4) {
        // std::cerr << "Error: tf2onnx model input dimension is not 3 or 4\n";
        // exit(1);
    }

    if (framework.compare("tf2onnx") == 0 && d.size() == 3) {
        channel = d.at(0).dim_value();  // it should be always 1?!
        input_height = d.at(1).dim_value();
        input_width = d.at(2).dim_value();
    } else if (framework.compare("tf2onnx") == 0 && d.size() == 4) {
        channel = d.at(3).dim_value();
        input_height = d.at(1).dim_value();
        input_width = d.at(2).dim_value();
    } else {  // framework pytorch
        channel = d.at(1).dim_value();
        input_height = d.at(2).dim_value();
        input_width = d.at(3).dim_value();
    }
    dim_t next_layer_dim{input_width, input_height, channel};

    bool first_dense = true;

    // for each layer (called node in onnx)
    for (auto node : graph.node()) {
        if (node.op_type().compare("Flatten") == 0 ||
            (node.op_type().compare("Reshape") == 0 &&
             node.output(0).find("flatten") != std::string::npos)) {
            layer.push_back(new Flatten(next_layer_dim));

        } else if (node.op_type().compare("Gemm") == 0) {
            // get input and output dimensions
            size_t input_dim;
            size_t output_dim;

            if (framework.compare("tf2onnx") == 0) {
                input_dim = wandb[wandb_tensor_cnt - 1].dims(0);
                output_dim = wandb[wandb_tensor_cnt - 1].dims(1);
            } else {
                input_dim = wandb[wandb_tensor_cnt].dims(1);
                output_dim = wandb[wandb_tensor_cnt].dims(0);
            }

            // get weights and biases
            vector<wandb_t> weights;
            vector<wandb_t> biases;
            ScalarTensor<wandb_t> weights_tensor;
            ScalarTensor<wandb_t> biases_tensor;
            Dense* dense;
            if (framework.compare("tf2onnx") == 0) {
                biases = parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt--;
                biases_tensor =
                    ScalarTensor<wandb_t>(biases, dim_t{output_dim});
                weights = parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt--;
                weights_tensor = ScalarTensor<wandb_t>(
                    weights, dim_t{input_dim, output_dim});
                weights_tensor.transpose();
                // Check whether there is a conv2d layer in front of a flatten
                // and get number of output-channels. Do this check only for the
                // first dense layer of the model.
                int channel = get_nr_channel_before_flatten(layer);
                // create layer
                if (channel > 0 && first_dense) {
                    first_dense = false;
                    dense = new Dense(weights_tensor, biases_tensor, q_method,
                                      q_const, channel);
                } else {
                    dense = new Dense(weights_tensor, biases_tensor, q_method,
                                      q_const);
                }
            } else {
                weights = parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt++;
                weights_tensor = ScalarTensor<wandb_t>(
                    weights, dim_t{output_dim, input_dim});
                biases = parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt++;
                biases_tensor =
                    ScalarTensor<wandb_t>(biases, dim_t{output_dim});
                // create layer
                dense =
                    new Dense(weights_tensor, biases_tensor, q_method, q_const);
            }

            layer.push_back(dense);
            if (q_method == QuantizationMethod::ScaleQuant) {
                Rescale* rescale;
#ifdef QS
                const vector<crt_val_t> s = {QS};//assumes that QS is part of CRT base
                rescale = new Rescale(s, dense->get_output_dims());
#else
                rescale = new Rescale(QL, dense->get_output_dims());
#endif
            layer.push_back(rescale);
            }
            next_layer_dim = dense->get_output_dims();
        } else if (node.op_type().compare("Conv") == 0) {
            // get convd2d parameters
            //// input dimensions
            size_t input_height = next_layer_dim.at(0);
            size_t input_width = next_layer_dim.at(1);
            size_t channel = next_layer_dim.at(2);

            //// kernel shape
            auto filter_shapes = node.attribute().at(2);
            size_t filter_height = filter_shapes.ints(0);
            size_t filter_width = filter_shapes.ints(1);

            //// stride
            size_t stride_height;
            size_t stride_width;
            if (framework.compare("tf2onnx") == 0) {
                auto strides = node.attribute().at(1);
                stride_height = strides.ints(0);
                stride_width = strides.ints(1);
            } else {
                auto strides = node.attribute().at(4);
                stride_height = strides.ints(0);
                stride_width = strides.ints(1);
            }

            //// paramter dimensions
            size_t filter = wandb[wandb_tensor_cnt].dims(0);
            // int filter_height = wandb[wandb_tensor_cnt].dims(2);
            // int filter_width = wandb[wandb_tensor_cnt].dims(3);

            ScalarTensor<wandb_t> biases_tensor;
            ScalarTensor<wandb_t> weights_tensor;
            if (framework.compare("tf2onnx") == 0) {
                auto biases =
                    parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt--;
                biases_tensor = ScalarTensor<wandb_t>(biases, dim_t{filter});
                auto weights =
                    parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt--;
                dim_t weights_dim{filter_width, filter_height, channel, filter};
                weights_tensor = ScalarTensor<wandb_t>(weights, weights_dim);
            } else {
                auto weights =
                    parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt++;
                dim_t weights_dim{filter_width, filter_height, channel, filter};
                weights_tensor = ScalarTensor<wandb_t>(weights, weights_dim);
                auto biases =
                    parse_raw_data(wandb[wandb_tensor_cnt].raw_data());
                wandb_tensor_cnt++;
                biases_tensor = ScalarTensor<wandb_t>(biases, dim_t{filter});
            }

            // create layer
            auto conv = new Conv2d{weights_tensor, biases_tensor, input_width,
                                   input_height,   channel,       filter,
                                   filter_width,   filter_height, stride_width,
                                   stride_height,  q_method,      q_const};

            layer.push_back(conv);
            if (q_method == QuantizationMethod::ScaleQuant) {
                Rescale* rescale;
#ifdef QS
                const vector<crt_val_t> s = {QS};//assumes that QS is part of CRT base
                rescale = new Rescale(s, conv->get_output_dims());
#else
                rescale = new Rescale(QL, conv->get_output_dims());
#endif
                layer.push_back(rescale);
            }
            next_layer_dim = conv->get_output_dims();
        } else if (node.op_type().compare("MaxPool") == 0) {
            size_t input_height = next_layer_dim.at(0);
            size_t input_width = next_layer_dim.at(1);
            size_t channel = next_layer_dim.at(2);
            size_t kernel_height = node.attribute().at(1).ints(0);
            size_t kernel_width = node.attribute().at(1).ints(1);
            size_t stride_height = node.attribute().at(3).ints(0);
            size_t stride_width = node.attribute().at(3).ints(1);

            auto pool =
                new MaxPool2d(input_width, input_height, channel, kernel_width,
                              kernel_height, stride_width, stride_height);
            layer.push_back(pool);
            next_layer_dim = pool->get_output_dims();
        } else if (node.op_type().compare("Relu") == 0) {
            auto relu = new Relu(next_layer_dim);
            layer.push_back(relu);
        } else if (node.op_type().compare("Tanh") == 0 ||
                   node.op_type().compare("SignTanh") == 0) {
            auto sign = new Sign(next_layer_dim);
            layer.push_back(sign);
        }
    }
    Circuit* circuit = new Circuit(layer);

    return circuit;
}

Circuit* load_onnx_model(
    const std::string& path,
    QuantizationMethod q_method = QuantizationMethod::SimpleQuant,
    wandb_t q_const = 0.02) {
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto model = parse_model_file(path);
    auto circuit = create_circuit_from_onnx_model(model, q_method, q_const);

    // Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return circuit;
}

#endif
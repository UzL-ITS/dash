#ifndef TEST_CONV2D_H
#define TEST_CONV2D_H

#include <numeric>
#include <vector>

#define QUANTIZATION_CONSTANT 10
#define QUANTIZATION_METHOD QuantizationMethod::SimpleQuant

#include "circuit/circuit.h"
#include "circuit/layer/conv2d.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit.h"
#include "garbling/layer/garbled_conv2d.h"
#include "misc/datatypes.h"
#include "misc/util.h"

using std::vector;

typedef struct conv2d_params {
    ScalarTensor<q_val_t> inputs;
    ScalarTensor<wandb_t> weights;
    ScalarTensor<wandb_t> biases;
    size_t input_width;
    size_t input_height;
    size_t channel;
    size_t filter;
    size_t filter_width;
    size_t filter_height;
    size_t stride_width;
    size_t stride_height;
} conv2d_params;

//TODO: add proper QL parametrization

// 2D Convolution as matrix multiplication:
// https://medium.com/@_init_/an-illustrated-explanation-of-performing-2d-convolutions-using-matrix-multiplications-1e8de8cd2544
// (Bias trick (https://cs231n.github.io/linear-classify/) is also possible)

// Explanation of dimension
// https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow

/**
 * @brief      2D convolution layer.
 *
 * Implements a subset of TensorFlow's Conv2D layer. Only supports
 * padding='valid'.
 *
 * TensorFlow semantic:
 * Input: number of images x height x width x channels
 * Filter: height x width x channels x number of filters
 *
 * @param      input          Pointer to the input of the layer.
 * @param      weights        Pointer to the weights of the layer.
 * @param      bias           Pointer to biases of the layer.
 * @param[in]  input_width    Width of the input.
 * @param[in]  input_height   Height of the input.
 * @param[in]  channel        Number of channels of the input.
 * @param[in]  filter         Number of filters.
 * @param[in]  filter_width   Width of the filters.
 * @param[in]  filter_height  Height of the filters.
 * @param[in]  stride_width   Stride width (along axis 1).
 * @param[in]  stride_height  Stride height (along axis 0).
 *
 * @return     Pointer to the output of the layer.
 */
vector<q_val_t> conv2d(conv2d_params cp) {
    int output_width = (cp.input_width - cp.filter_width) / cp.stride_width + 1;
    int output_height =
        (cp.input_height - cp.filter_height) / cp.stride_height + 1;

    int output_size = output_width * output_height;  // sizer per filter
    int filter_size = cp.filter_width * cp.filter_height;
    int input_size = cp.input_width * cp.input_height;

    vector<q_val_t> output(output_size * cp.filter);

    // for all filter
    // #pragma omp parallel for collapse(3)
    for (int v = 0; v < cp.filter; ++v) {
        // move filter along y-axis
        for (int l = 0; l < output_height; ++l) {
            // move filter along x-axis
            for (int k = 0; k < output_width; ++k) {
                // scalar product over all input-channel...
                for (int w = 0; w < cp.channel; ++w) {
                    // ...along filter height and...
                    for (int i = 0; i < cp.filter_height; ++i) {
                        // ...width
                        for (int j = 0; j < cp.filter_width; ++j) {
                            output[v * output_size + l * output_width + k] +=
                                std::llround(
                                    cp.weights[v * filter_size * cp.channel +
                                               w * filter_size +
                                               i * cp.filter_height + j] /
                                    QUANTIZATION_CONSTANT) *
                                cp.inputs[w * input_size + i * cp.input_width +
                                          j + k * cp.stride_width +
                                          l * cp.stride_height *
                                              cp.input_width];
                        }
                    }
                }
                // add bias to each output-channel
                output[v * output_size + l * output_width + k] +=
                    std::llround(cp.biases[v] / QUANTIZATION_CONSTANT);
            }
        }
    }
    return output;
}

class TestSingleConv2d : public ::testing::TestWithParam<conv2d_params> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto conv2d{new Conv2d{
            GetParam().weights, GetParam().biases, GetParam().input_width,
            GetParam().input_height, GetParam().channel, GetParam().filter,
            GetParam().filter_width, GetParam().filter_height,
            GetParam().stride_width, GetParam().stride_height,
            5, QUANTIZATION_METHOD, QUANTIZATION_CONSTANT}};
        m_circuit = new Circuit{conv2d};
        m_gc = new GarbledCircuit{m_circuit, 8};
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

TEST_P(TestSingleConv2d, EndtoEndCPU) {
    auto cp = GetParam();

    auto g_inputs{m_gc->garble_inputs(cp.inputs)};
    auto g_outputs{m_gc->cpu_evaluate(g_inputs)};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    vector<q_val_t> expected_outputs{conv2d(cp)};

    EXPECT_EQ(outputs.as_vector(), expected_outputs);

    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

// TODO: re-enable after bringing eigen implementations to GPU
// TEST_P(TestSingleConv2d, EndtoEndGPU) {
//     auto cp = GetParam();
//     m_gc->cuda_move();
//     auto g_inputs{m_gc->garble_inputs(cp.inputs)};
//     auto g_dev_inputs{m_gc->cuda_move_inputs(g_inputs)};
//     m_gc->cuda_evaluate(g_dev_inputs);
//     auto g_outputs{m_gc->cuda_move_outputs()};
//     auto outputs{m_gc->decode_outputs(g_outputs)};

//     vector<q_val_t> expected_outputs{conv2d(cp)};

//     EXPECT_EQ(outputs.as_vector(), expected_outputs);

//     for (auto label : *g_inputs) {
//         delete label;
//     }
//     delete g_inputs;
//     m_gc->cuda_free_inputs(g_dev_inputs);
// }

// TEST(TestTwoConv2d, EndtoEndGPU) {
//     // Layer parameters
//     // - first conv2d
//     size_t input_width = 16;
//     size_t input_height = 16;
//     size_t channel = 3;
//     size_t filter = 2;
//     size_t filter_width = 4;
//     size_t filter_height = 4;
//     size_t stride_width = 2;
//     size_t stride_height = 2;

//     dim_t input_dims{input_width, input_height, channel};
//     auto inputs{util::get_random_vector<q_val_t>(
//         input_width * input_height * channel, 0, 10)};
//     ScalarTensor<q_val_t> input_tensor{inputs, input_dims};

//     dim_t weight_dims{filter_width, filter_height, channel, filter};
//     vector<wandb_t> weights{util::get_random_vector<wandb_t>(
//         filter_width * filter_height * channel * filter, 0, 10)};
//     ScalarTensor<wandb_t> weight_tensor{weights, weight_dims};

//     dim_t bias_dims{filter};
//     auto biases{util::get_random_vector<wandb_t>(filter, 0, 10)};
//     ScalarTensor<wandb_t> bias_tensor{biases, bias_dims};

//     size_t input_width2 = (input_width - filter_width) / stride_width + 1;
//     size_t input_height2 = (input_height - filter_height) / stride_height + 1;
//     size_t channel2 = filter;
//     size_t filter2 = 2;
//     size_t filter_width2 = 3;
//     size_t filter_height2 = 3;
//     size_t stride_width2 = 1;
//     size_t stride_height2 = 1;

//     // - second conv2d
//     dim_t weights2_dims{3, 3, 2, 2};
//     auto weights2{util::get_random_vector<wandb_t>(
//         filter_width2 * filter_height2 * channel2 * filter2, 0, 100)};
//     ScalarTensor<wandb_t> weight_tensor2{weights2, weights2_dims};

//     dim_t biases2_dims{filter2};
//     auto biases2{util::get_random_vector<wandb_t>(filter2, 0, 100)};
//     ScalarTensor<wandb_t> bias_tensor2{biases2, biases2_dims};

//     // construct circuit
//     auto conv1{new Conv2d{weight_tensor, bias_tensor, input_width, input_height,
//                           channel, filter, filter_width, filter_height,
//                           stride_width, stride_height, QUANTIZATION_METHOD,
//                           5, QUANTIZATION_CONSTANT}};

//     auto conv2{new Conv2d{weight_tensor2, bias_tensor2, input_width2,
//                           input_height2, channel2, filter2, filter_width2,
//                           filter_height2, stride_width2, stride_height2,
//                           5, QUANTIZATION_METHOD, QUANTIZATION_CONSTANT}};

//     auto circuit = new Circuit{conv1, conv2};
//     auto gc = new GarbledCircuit(circuit, 8);
//     gc->cuda_move();
//     auto g_inputs{gc->garble_inputs(input_tensor)};
//     auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//     gc->cuda_evaluate(g_dev_inputs);
//     auto g_outputs{gc->cuda_move_outputs()};
//     auto outputs{gc->decode_outputs(g_outputs)};

//     // compute expected outputs
//     conv2d_params cp{input_tensor,  weight_tensor, bias_tensor,  input_width,
//                      input_height,  channel,       filter,       filter_width,
//                      filter_height, stride_width,  stride_height};
//     auto out1 = conv2d(cp);

//     auto out1_tensor{
//         ScalarTensor<q_val_t>{out1, {input_width2, input_height2, channel2}}};
//     conv2d_params cp2{out1_tensor,   weight_tensor2, bias_tensor2,
//                       input_width2,  input_height2,  channel2,
//                       filter2,       filter_width2,  filter_height2,
//                       stride_width2, stride_height2};
//     auto out2 = conv2d(cp2);

//     EXPECT_EQ(outputs.as_vector(), out2);

//     // clean up
//     for (auto label : *g_inputs) {
//         delete label;
//     }
//     delete g_inputs;
//     gc->cuda_free_inputs(g_dev_inputs);

//     delete circuit;
//     delete gc;
// }

// dims: input_width x input_height x channel
ScalarTensor<q_val_t> input1{{0, 1, 2, 3, 4, 5, 6, 7, 8}, dim_t{3, 3, 1}};
vector<q_val_t> v2{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 9, 6, 12, 2};
ScalarTensor<q_val_t> input2{2, dim_t{3, 3, 2}};
vector<q_val_t> v3{0, 1, 2, 3, 4, 5, 6,  7, 8, 9, 10, 11, 12, 13, 14, 15,
                   0, 1, 4, 3, 4, 5, 11, 7, 8, 9, 3,  11, 7,  13, 9,  15};
ScalarTensor<q_val_t> input3{v3, dim_t{4, 4, 2}};

// identity function
conv2d_params cp0 = {.inputs = input1,
                     // dims: filter_width x filter_height x channel x filter
                     .weights = ScalarTensor<wandb_t>{10, dim_t{1, 1, 1, 1}},
                     // dims: filter
                     .biases = ScalarTensor<wandb_t>{10, dim_t{1}},
                     .input_width = 3,
                     .input_height = 3,
                     .channel = 1,
                     .filter = 1,
                     .filter_width = 1,
                     .filter_height = 1,
                     .stride_width = 1,
                     .stride_height = 1};

// filter size greater 1
conv2d_params cp1 = {
    .inputs = input1,
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40}, dim_t{2, 2, 1, 1}},
    .biases = ScalarTensor<wandb_t>{10, dim_t{1}},
    .input_width = 3,
    .input_height = 3,
    .channel = 1,
    .filter = 1,
    .filter_width = 2,
    .filter_height = 2,
    .stride_width = 1,
    .stride_height = 1};

// multiple input channel
conv2d_params cp2 = {
    .inputs = input2,
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40, 50, 60, 70, 80},
                                     dim_t{2, 2, 2, 1}},
    .biases = ScalarTensor<wandb_t>{10, dim_t{1}},
    .input_width = 3,
    .input_height = 3,
    .channel = 2,
    .filter = 1,
    .filter_width = 2,
    .filter_height = 2,
    .stride_width = 1,
    .stride_height = 1};

// multiple input channel and filter
conv2d_params cp3 = {
    .inputs = input2,
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                      110, 120, 130, 140, 150, 160},
                                     dim_t{2, 2, 2, 2}},
    .biases = ScalarTensor<wandb_t>{{10, 20}, dim_t{2}},
    .input_width = 3,
    .input_height = 3,
    .channel = 2,
    .filter = 2,
    .filter_width = 2,
    .filter_height = 2,
    .stride_width = 1,
    .stride_height = 1};

// multiple input channel and filter and stride greater 1
conv2d_params cp4 = {
    .inputs = input3,
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                      110, 120, 130, 140, 150, 160},
                                     dim_t{2, 2, 2, 2}},
    .biases = ScalarTensor<wandb_t>{{10, 20}, dim_t{2}},
    .input_width = 4,
    .input_height = 4,
    .channel = 2,
    .filter = 2,
    .filter_width = 2,
    .filter_height = 2,
    .stride_width = 2,
    .stride_height = 2};

vector<conv2d_params> params{cp0, cp1, cp2, cp3, cp4};

INSTANTIATE_TEST_SUITE_P(TestSingleConv2d, TestSingleConv2d,
                         ::testing::ValuesIn(params));

#endif
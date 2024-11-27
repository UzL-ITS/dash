#ifndef TEST_ONNX_MODEL_LOADER_H
#define TEST_ONNX_MODEL_LOADER_H

#include "circuit/circuit.h"
#include "circuit/onnx_modelloader.h"
#include "misc/datatypes.h"

class TestOnnxModelLoader : public ::testing::Test {};

TEST(TestOnnxModelLoader, Dense) {
    Circuit* circuit = load_onnx_model("fixtures/model_dense.onnx",
                                       QuantizationMethod::SimpleQuant, 10);
    dim_t expected_dim{28, 28, 1};
    EXPECT_EQ(circuit->get_input_dims(), expected_dim);
    EXPECT_EQ(circuit->get_output_dims().at(0), 10);

    // Check first weight
    float weight = -0.02056167833507061004638671875;
    // The first layer is a flatte layer
    Dense* layer = dynamic_cast<Dense*>(circuit->get_layer()[1]);
    EXPECT_EQ(layer->get_weights().at(0), weight);

    // Check 10th weight
    weight = 0.00375685538165271282196044921875;
    EXPECT_EQ(layer->get_weights().at(9), weight);

    // Check last weight
    weight = 0.018187098205089569091796875;
    EXPECT_EQ(layer->get_weights().back(), weight);

    // Check first bias
    weight = -0.02799860946834087371826171875;
    EXPECT_EQ(layer->get_biases().at(0), weight);

    // Check 5th bias
    weight = -0.0069468836300075054168701171875;
    EXPECT_EQ(layer->get_biases().at(4), weight);

    // Check last bias
    weight = -0.02649968676269054412841796875;
    EXPECT_EQ(layer->get_biases().back(), weight);
    delete circuit;
}

#endif
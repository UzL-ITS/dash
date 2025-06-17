#ifndef TEST_MAXPOOL2D_H
#define TEST_MAXPOOL2D_H

#include <vector>

#include "circuit/circuit.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/max_pool2d.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit.h"
#include "misc/datatypes.h"

using std::vector;

struct maxpool2d_test_t {
    ScalarTensor<q_val_t> inputs;
    vector<crt_val_t> crt_base;
    vector<mrs_val_t> mrs_base;
};

class TestGarbledMaxPool2D : public ::testing::TestWithParam<maxpool2d_test_t> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto input_dims = GetParam().inputs.get_dims();
        m_circuit = new Circuit{new MaxPool2d{input_dims.at(0), input_dims.at(1), input_dims.at(2), 2, 2}};
        m_gc = new GarbledCircuit(m_circuit, GetParam().crt_base,
                                  GetParam().mrs_base);
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

class TestGarbledMaxPool2DMultKernel : public ::testing::TestWithParam<maxpool2d_test_t> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto input_dims = GetParam().inputs.get_dims();
        m_circuit = new Circuit{new MaxPool2d{input_dims.at(0), input_dims.at(1), input_dims.at(2), 1, 1}};
        m_gc = new GarbledCircuit(m_circuit, GetParam().crt_base,
                                  GetParam().mrs_base);
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

//Note: assumes 2x2 kernel, 1x1 stride, 1x1 padding and 2x2 input
TEST_P(TestGarbledMaxPool2D, MaxPool2DCPU) {
    auto inputs = GetParam().inputs;
    auto g_inputs = m_gc->garble_inputs(inputs);
    auto g_outputs = m_gc->cpu_evaluate(g_inputs);
    auto outputs = m_gc->decode_outputs(g_outputs);

    auto input_vector = inputs.as_vector();
    const q_val_t expected = *std::max_element(input_vector.begin(), input_vector.end());

    EXPECT_EQ(outputs.as_vector(), vector<q_val_t>{expected});

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

TEST_P(TestGarbledMaxPool2DMultKernel, MaxPool2DCPU) {
    auto inputs = GetParam().inputs;
    auto g_inputs = m_gc->garble_inputs(inputs);
    auto g_outputs = m_gc->cpu_evaluate(g_inputs);
    auto outputs = m_gc->decode_outputs(g_outputs);

    auto input_vector = inputs.as_vector();
    // Here, we test multiple 1x1 kernels. The operation should return the input tensor.

    EXPECT_EQ(outputs.as_vector(), input_vector);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

namespace test_garbled_maxpool2d {
maxpool2d_test_t zero{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{0, 0, 0, 0}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{1, 1, 1, 1}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t neg_low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-1, -1, -1, -1}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t asc_low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-1, 0, 1, 2}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t desc_low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-2, -1, 0, 1}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t mixed_low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-1, -2, 0, 2}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{7, 7, 7, 7}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t neg_mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-7, -7, -7, -7}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t asc_mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-7, -4, 3, 6}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t desc_mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{7, 4, -3, -6}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t mixed_mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-5, 3, 0, 4}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{14, 14, 14, 14}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t neg_high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-15, -15, -15, -15}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t asc_high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-14, -7, 0, 14}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t desc_high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-104, 1, 0, 7}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7},
    .mrs_base = vector<mrs_val_t>{26, 3}};

maxpool2d_test_t mixed_high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{3, -11, 5, 10}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

maxpool2d_test_t zero_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{0, 0, 0, 0}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

maxpool2d_test_t low_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{1, 1, 1, 1}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

maxpool2d_test_t neg_low_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-1, -1, -1, -1}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

maxpool2d_test_t mid_p9{
    .inputs =
        ScalarTensor<q_val_t>{vector<q_val_t>{55773217, 55773217, 55773217, 55773217}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

maxpool2d_test_t neg_mid_p9{
    .inputs =
        ScalarTensor<q_val_t>{vector<q_val_t>{-55773217, -55773217, -55773217, -55773217}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

maxpool2d_test_t high_p9{
    .inputs =
        ScalarTensor<q_val_t>{vector<q_val_t>{111546434, 111546434, 111546434, 111546434}, dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

maxpool2d_test_t neg_high_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-111546435, -111546435, -111546435, -111546435},
                                    dim_t{2, 2, 1}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

vector<maxpool2d_test_t> test_values = {
    zero,
    low, neg_low, asc_low, desc_low, mixed_low,
    mid, neg_mid, asc_mid, desc_mid, mixed_mid,
    high, neg_high, asc_high, desc_high, mixed_high,
    high_p9, neg_high_p9, zero_p9, low_p9, neg_low_p9, mid_p9, neg_mid_p9};
}  // namespace test_garbled_maxpool2d

INSTANTIATE_TEST_CASE_P(TestMaxPool2DLayer, TestGarbledMaxPool2D,
                        ::testing::ValuesIn(test_garbled_maxpool2d::test_values));
INSTANTIATE_TEST_CASE_P(TestMaxPool2DLayerMultKernel, TestGarbledMaxPool2DMultKernel,
                        ::testing::ValuesIn(test_garbled_maxpool2d::test_values));

#endif
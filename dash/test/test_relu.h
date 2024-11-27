#ifndef TEST_RELU_H
#define TEST_RELU_H

#include <vector>

#include "circuit/circuit.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/relu.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit.h"
#include "misc/datatypes.h"

using std::vector;

struct relu_test_t {
    ScalarTensor<q_val_t> inputs;
    vector<crt_val_t> crt_base;
    vector<mrs_val_t> mrs_base;
};

class TestGarbledRelu : public ::testing::TestWithParam<relu_test_t> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto input_dims = GetParam().inputs.get_dims();
        auto output_dims = input_dims;
        m_circuit = new Circuit{new Relu{dim_t{2}}};
        m_gc = new GarbledCircuit(m_circuit, GetParam().crt_base,
                                  GetParam().mrs_base);
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

class TestGarbledReluMultLayer : public ::testing::TestWithParam<relu_test_t> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto input_dims = GetParam().inputs.get_dims();
        auto output_dims = input_dims;
        m_circuit = new Circuit{
            new Dense{ScalarTensor<wandb_t>{{1, 0, 1, 0}, dim_t{2, 2}},
                      ScalarTensor<wandb_t>{{0, 0}, dim_t{2}}},
            new Relu{dim_t{2}},
            new Dense{ScalarTensor<wandb_t>{{1, 0, 1, 0}, dim_t{2, 2}},
                      ScalarTensor<wandb_t>{{0, 0}, dim_t{2}}}};
        m_gc = new GarbledCircuit(m_circuit, GetParam().crt_base,
                                  GetParam().mrs_base);
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

q_val_t relu_func(q_val_t x) { return x > 0 ? x : 0; }

TEST_P(TestGarbledRelu, ReluCPU) {
    auto inputs = GetParam().inputs;
    auto g_inputs = m_gc->garble_inputs(inputs);
    auto g_outputs = m_gc->cpu_evaluate(g_inputs);
    auto outputs = m_gc->decode_outputs(g_outputs);

    auto expected{GetParam().inputs};
    expected.map(&relu_func);

    EXPECT_EQ(outputs.as_vector(), expected.as_vector());

    // clen up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

TEST_P(TestGarbledReluMultLayer, ReluCPU) {
    auto inputs = GetParam().inputs;
    auto g_inputs = m_gc->garble_inputs(inputs);
    auto g_outputs = m_gc->cpu_evaluate(g_inputs);
    auto outputs = m_gc->decode_outputs(g_outputs);

    auto expected{GetParam().inputs};
    expected.map(&relu_func);

    EXPECT_EQ(outputs.as_vector(), expected.as_vector());

    // clen up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

TEST_P(TestGarbledRelu, ReluGPU) {
    auto inputs = GetParam().inputs;
    m_gc->cuda_move();
    auto g_inputs = m_gc->garble_inputs(inputs);
    auto g_dev_inputs{m_gc->cuda_move_inputs(g_inputs)};
    m_gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs = m_gc->cuda_move_outputs();
    auto outputs = m_gc->decode_outputs(g_outputs);

    auto expected{GetParam().inputs};
    expected.map(&relu_func);

    EXPECT_EQ(outputs.as_vector(), expected.as_vector());

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    m_gc->cuda_free_inputs(g_dev_inputs);
    delete g_inputs;
}

TEST_P(TestGarbledReluMultLayer, ReluGPU) {
    auto inputs = GetParam().inputs;
    m_gc->cuda_move();
    auto g_inputs = m_gc->garble_inputs(inputs);
    auto g_dev_inputs{m_gc->cuda_move_inputs(g_inputs)};
    m_gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs = m_gc->cuda_move_outputs();
    auto outputs = m_gc->decode_outputs(g_outputs);

    auto expected{GetParam().inputs};
    expected.map(&relu_func);

    EXPECT_EQ(outputs.as_vector(), expected.as_vector());

    // clen up
    for (auto label : *g_inputs) {
        delete label;
    }
    m_gc->cuda_free_inputs(g_dev_inputs);
    delete g_inputs;
}

namespace test_garbled_relu {
relu_test_t zero{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{0, 0}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{1, 1}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t neg_low{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-1, -1}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{7, 7}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t neg_mid{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-7, -7}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{14, 14}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t neg_high{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-15, -15}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5},
    .mrs_base = vector<mrs_val_t>{26, 6, 3, 2}};

relu_test_t zero_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{0, 0}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

relu_test_t low_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{1, 1}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

relu_test_t neg_low_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-1, -1}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

relu_test_t mid_p9{
    .inputs =
        ScalarTensor<q_val_t>{vector<q_val_t>{55773217, 55773217}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

relu_test_t neg_mid_p9{
    .inputs =
        ScalarTensor<q_val_t>{vector<q_val_t>{-55773217, -55773217}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

relu_test_t high_p9{
    .inputs =
        ScalarTensor<q_val_t>{vector<q_val_t>{111546434, 111546434}, dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

relu_test_t neg_high_p9{
    .inputs = ScalarTensor<q_val_t>{vector<q_val_t>{-111546435, -111546435},
                                    dim_t{2}},
    .crt_base = vector<crt_val_t>{2, 3, 5, 7, 11, 13, 17, 19, 23},
    .mrs_base = vector<mrs_val_t>{76, 7, 7, 7, 7, 7, 5, 5}};

vector<relu_test_t> test_values = {
    zero,    low,         neg_low, mid,    neg_mid,    high,   neg_high,
    high_p9, neg_high_p9, zero_p9, low_p9, neg_low_p9, mid_p9, neg_mid_p9};
}  // namespace test_garbled_relu

INSTANTIATE_TEST_CASE_P(TestReluLayer, TestGarbledRelu,
                        ::testing::ValuesIn(test_garbled_relu::test_values));

#endif
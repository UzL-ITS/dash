#ifndef MIXED_MOD_MULT_H
#define MIXED_MOD_MULT_H

#include <vector>

#include "circuit/circuit.h"
#include "misc/datatypes.h"
#include "garbling/garbled_circuit.h"
#include "circuit/layer/mixed_mod_mult_layer.h"
#include "circuit/scalar_tensor.h"

using std::vector;

class TestMixedModMult : public ::testing::TestWithParam<vector<q_val_t>> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        dim_t input_dims{2};
        dim_t output_dims{1};
        m_circuit =
            new Circuit{new MixedModMultLayer{input_dims, output_dims, 2}};
        vector<crt_val_t> crt_base{19};
        m_gc = new GarbledCircuit(m_circuit, crt_base);
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

TEST_P(TestMixedModMult, Mult) {
    ScalarTensor<q_val_t> inputs{GetParam(), dim_t{2}};

    auto g_inputs{m_gc->garble_inputs(inputs)};
    auto g_outputs{m_gc->cpu_evaluate(g_inputs)};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    q_val_t expected = std::accumulate(GetParam().begin(), GetParam().end(), 1,
                                       std::multiplies<q_val_t>());

    EXPECT_EQ(outputs.data()[0], expected);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

TEST_P(TestMixedModMult, MultGPU) {
    ScalarTensor<q_val_t> inputs{GetParam(), dim_t{2}};
    m_gc->cuda_move();
    auto g_inputs{m_gc->garble_inputs(inputs)};
    auto g_dev_inputs{m_gc->cuda_move_inputs(g_inputs)};
    m_gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{m_gc->cuda_move_outputs()};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    q_val_t expected = std::accumulate(GetParam().begin(), GetParam().end(), 1,
                                       std::multiplies<q_val_t>());

    EXPECT_EQ(outputs.data()[0], expected);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    m_gc->cuda_free_inputs(g_dev_inputs);
}

namespace test_mixed_mod_mult {
vector<q_val_t> zeros{0, 0};
vector<q_val_t> zero_pos{1, 0};
vector<q_val_t> zero_neg{-1, 0};
vector<q_val_t> low_pospos{2, 1};
vector<q_val_t> mid_pospos{4, 1};
vector<q_val_t> high_pospos{8, 1};
vector<q_val_t> low_negpos{-2, 1};
vector<q_val_t> mid_negpos{-4, 1};
vector<q_val_t> high_negpos{-8, 1};

vector<vector<q_val_t>> test_values = {zeros,      zero_pos,   zero_neg,
                                       low_pospos, mid_pospos, high_pospos,
                                       low_negpos, mid_negpos, high_negpos};

}  // namespace test_mixed_mod_mult

INSTANTIATE_TEST_CASE_P(TestMixedModMultLayer, TestMixedModMult,
                        ::testing::ValuesIn(test_mixed_mod_mult::test_values));

#endif
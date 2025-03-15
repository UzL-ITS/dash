#ifndef TEST_DENSE_H
#define TEST_DENSE_H

#include <cuda_runtime_api.h>

#include <vector>

#define QUANTIZATION_CONSTANT 10
#define QUANTIZATION_METHOD QuantizationMethod::SimpleQuant

#include "circuit/circuit.h"
#include "circuit/layer/dense.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit.h"
#include "garbling/layer/garbled_dense.h"
#include "misc/util.h"

using std::vector;

typedef struct dense_params {
    ScalarTensor<q_val_t> inputs;
    ScalarTensor<wandb_t> weights;
    ScalarTensor<wandb_t> biases;
} dense_params;

class TestSingleDense : public ::testing::TestWithParam<dense_params> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto dense1{new Dense{GetParam().weights, GetParam().biases, 5,
                              QUANTIZATION_METHOD, QUANTIZATION_CONSTANT}};
        m_circuit = new Circuit{dense1};
        m_gc = new GarbledCircuit{m_circuit, 8};
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

class TestTwoDense : public ::testing::TestWithParam<dense_params> {
   protected:
    Circuit* m_circuit;
    GarbledCircuit* m_gc;
    void SetUp() override {
        auto dense1{new Dense{GetParam().weights, GetParam().biases, 5,
                              QUANTIZATION_METHOD, QUANTIZATION_CONSTANT}};
        auto dense2{new Dense{GetParam().weights, GetParam().biases, 5,
                              QUANTIZATION_METHOD, QUANTIZATION_CONSTANT}};
        m_circuit = new Circuit{dense1, dense2};
        m_gc = new GarbledCircuit{m_circuit, 8};
    }

    void TearDown() override {
        delete m_circuit;
        delete m_gc;
    }
};

ScalarTensor<q_val_t> cpu_matmul(dense_params dp) {
    size_t input_dim = dp.inputs.get_dims().at(0);
    size_t output_dim = dp.weights.get_dims().at(0);
    ScalarTensor<q_val_t> expected{dim_t{output_dim}};
    memset(expected.data(), '\0', output_dim * sizeof(q_val_t));

    for (int i = 0; i < output_dim; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            expected.data()[i] +=
                dp.inputs[j] * std::llround((dp.weights[i * input_dim + j] /
                                             QUANTIZATION_CONSTANT));
        }
        expected.data()[i] +=
            std::llround((dp.biases[i] / QUANTIZATION_CONSTANT));
    }
    return expected;
}

TEST_P(TestSingleDense, EndtoEndCPU) {
    auto dp = GetParam();
    auto g_inputs{m_gc->garble_inputs(dp.inputs)};
    auto g_outputs{m_gc->cpu_evaluate(g_inputs)};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    // auto expected{cpu_dense(dp, m_gc)};
    // reconstruct_sign_bit(expected, m_gc);
    auto expected{cpu_matmul(dp)};
    vector<q_val_t> expected_vec{expected.data(),
                                 expected.data() + expected.get_dims().at(0)};

    EXPECT_EQ(expected_vec, outputs.as_vector());

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

TEST_P(TestTwoDense, EndtoEndCPU) {
    auto dp = GetParam();
    auto g_inputs{m_gc->garble_inputs(dp.inputs)};
    auto g_outputs{m_gc->cpu_evaluate(g_inputs)};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    // auto expected{cpu_dense(dp, m_gc)};
    // reconstruct_sign_bit(expected, m_gc);
    auto expected{cpu_matmul(dp)};
    dense_params dp2{dp};
    // dp2.inputs = expected;
    memcpy(dp2.inputs.data(), expected.data(),
           expected.get_dims().at(0) * sizeof(q_val_t));
    auto expected2{cpu_matmul(dp2)};

    vector<q_val_t> expected2_vec{
        expected2.data(), expected2.data() + expected2.get_dims().at(0)};

    EXPECT_EQ(expected2_vec, outputs.as_vector());

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
}

#ifndef LABEL_TENSOR_USE_EIGEN //TODO: remove macro when Eigen operations in LT support CUDA
TEST_P(TestSingleDense, CudaMove) {
    m_gc->cuda_move();
    auto g_dense{static_cast<GarbledDense*>(m_gc->get_garbled_layer().at(0))};

    // // Check weights
    auto expected_w{g_dense->get_qe_weights()};
    auto expected_w_vec = vector<q_val_t>(
        expected_w.data(), expected_w.data() + expected_w.size());
    auto actual_w_dev{g_dense->get_dev_qe_weights()};
    vector<q_val_t> actual_w;
    actual_w.resize(expected_w.size());
    cudaCheckError(cudaMemcpy(actual_w.data(), actual_w_dev,
                              sizeof(q_val_t) * expected_w.size(),
                              cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected_w_vec, actual_w);

    // Check biases
    auto expected_b = g_dense->get_qe_bias_label();
    auto actual_b_dev = g_dense->get_dev_qe_bias_label();
    // crt_base_size x output_dim x nr_components
    vector<vector<vector<crt_val_t>>> actual_b;
    actual_b.resize(m_gc->get_crt_base().size());
    for (int i = 0; i < m_gc->get_crt_base().size(); ++i) {
        int nr_comps = LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
        actual_b.at(i).resize(g_dense->get_layer()->get_output_dims().at(0));
        for (int j = 0; j < g_dense->get_layer()->get_output_dims().at(0);
             ++j) {
            actual_b.at(i).at(j).resize(nr_comps);
            cudaCheckError(cudaMemcpy(
                actual_b.at(i).at(j).data(), actual_b_dev[i] + j * nr_comps,
                sizeof(crt_val_t) * nr_comps, cudaMemcpyDeviceToHost));
        }
    }
    for (int i = 0; i < m_gc->get_crt_base().size(); ++i) {
        for (int j = 0; j < g_dense->get_layer()->get_output_dims().at(0);
             ++j) {
            EXPECT_EQ(expected_b->at(i)->get_components_vec(j),
                      actual_b.at(i).at(j));
        }
    }
}

TEST_P(TestSingleDense, EndtoEndGPU) {
    m_gc->cuda_move();

    auto dp = GetParam();
    auto g_inputs{m_gc->garble_inputs(dp.inputs)};
    auto g_dev_inputs{m_gc->cuda_move_inputs(g_inputs)};
    m_gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{m_gc->cuda_move_outputs()};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    // auto expected{cpu_dense(dp, m_gc)};
    // reconstruct_sign_bit(expected, m_gc);
    auto expected{cpu_matmul(dp)};

    vector<q_val_t> expected_vec{expected.data(),
                                 expected.data() + expected.get_dims().at(0)};

    EXPECT_EQ(expected_vec, outputs.as_vector());

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    m_gc->cuda_free_inputs(g_dev_inputs);
}

TEST_P(TestTwoDense, EndtoEndGPU) {
    m_gc->cuda_move();

    auto dp = GetParam();
    auto g_inputs{m_gc->garble_inputs(dp.inputs)};
    auto g_dev_inputs{m_gc->cuda_move_inputs(g_inputs)};
    m_gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{m_gc->cuda_move_outputs()};
    auto outputs{m_gc->decode_outputs(g_outputs)};

    auto expected{cpu_matmul(dp)};
    dense_params dp2{dp};
    // dp2.inputs = expected;
    memcpy(dp2.inputs.data(), expected.data(),
           expected.get_dims().at(0) * sizeof(q_val_t));
    auto expected2{cpu_matmul(dp2)};

    vector<q_val_t> expected2_vec{
        expected2.data(), expected2.data() + expected2.get_dims().at(0)};

    EXPECT_EQ(expected2_vec, outputs.as_vector());

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    m_gc->cuda_free_inputs(g_dev_inputs);
}
#endif // LABEL_TENSOR_USE_EIGEN

namespace test_dense_layer {
dense_params dp0 = {
    .inputs = ScalarTensor<q_val_t>{{1, 2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40}, dim_t{2, 2}},
    .biases = ScalarTensor<wandb_t>{{50, 60}, dim_t{2}},
};

dense_params dp1 = {
    .inputs = ScalarTensor<q_val_t>{{1, 2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40, 50, 60}, dim_t{3, 2}},
    .biases = ScalarTensor<wandb_t>{{50, 60, 70}, dim_t{3}},
};

dense_params dp2 = {
    .inputs = ScalarTensor<q_val_t>{{1, 2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{10, 20}, dim_t{1, 2}},
    .biases = ScalarTensor<wandb_t>{{50}, dim_t{1}},
};

dense_params dp3 = {
    .inputs = ScalarTensor<q_val_t>{{-1, -2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40}, dim_t{2, 2}},
    .biases = ScalarTensor<wandb_t>{{50, 60}, dim_t{2}},
};

dense_params dp4 = {
    .inputs = ScalarTensor<q_val_t>{{1, 2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{-10, -20, -30, -40}, dim_t{2, 2}},
    .biases = ScalarTensor<wandb_t>{{50, 60}, dim_t{2}},
};

dense_params dp5 = {
    .inputs = ScalarTensor<q_val_t>{{1, 2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{10, 20, 30, 40}, dim_t{2, 2}},
    .biases = ScalarTensor<wandb_t>{{-50, -60}, dim_t{2}},
};

dense_params dp6 = {
    .inputs = ScalarTensor<q_val_t>{{1, -2}, dim_t{2}},
    .weights = ScalarTensor<wandb_t>{{10, -20, -30, 40}, dim_t{2, 2}},
    .biases = ScalarTensor<wandb_t>{{-50, 60}, dim_t{2}},
};

dense_params dp7 = {
    .inputs =
        ScalarTensor<q_val_t>{util::get_random_vector<long long>(100, -5, 5),
                              dim_t{100}},
    .weights = ScalarTensor<wandb_t>{util::get_random_vector<wandb_t>(50 * 100,
                                                                      -50, 50),
                                     dim_t{50, 100}},
    .biases =
        ScalarTensor<wandb_t>{util::get_random_vector<wandb_t>(50, -50, 50),
                              dim_t{50}},
};
vector<dense_params> params1 = {dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7};
vector<dense_params> params2 = {dp0, dp3, dp4, dp5, dp6};
}  // namespace test_dense_layer

INSTANTIATE_TEST_SUITE_P(TestDenseLayer, TestSingleDense,
                         ::testing::ValuesIn(test_dense_layer::params1));

INSTANTIATE_TEST_SUITE_P(TestDenseLayer, TestTwoDense,
                         ::testing::ValuesIn(test_dense_layer::params2));

#endif
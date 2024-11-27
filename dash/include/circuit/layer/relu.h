#ifndef RELU_H
#define RELU_H

#include <algorithm>
#include <cstdlib>

#include "circuit/scalar_tensor.h"
#include "layer.h"
#include "misc/datatypes.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

template <typename T>
T relu(T x) {
    return x > 0 ? x : 0;
}

class Relu : public Layer {
   public:
    Relu(dim_t input_dims) : Layer(input_dims, input_dims) {}

    LayerType get_type() const override { return LayerType::approx_relu; }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        ScalarTensor<wandb_t> output(input.get_dims());
        for (size_t i = 0; i < input.size(); i++) {
            output.push_back(relu(input[i]));
        }
        if (track_extreme_values) {
            m_min_plain_val = std::min(m_min_plain_val, output.min());
            m_max_plain_val = std::max(m_max_plain_val, output.max());
        }
        return output;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override {
        ScalarTensor<q_val_t> output(input.get_dims());
        for (size_t i = 0; i < input.size(); i++) {
            output.push_back(relu(input[i]));
        }
        if (track_extreme_values) {
            m_min_plain_q_val = std::min(m_min_plain_q_val, input.min());
            m_max_plain_q_val = std::max(m_max_plain_q_val, input.max());
            m_min_plain_q_val = std::min(m_min_plain_q_val, output.min());
            m_max_plain_q_val = std::max(m_max_plain_q_val, output.max());
        }

        return output;
    }

    void quantize(wandb_t q_const) override {
        m_min_plain_q_val = Q_VAL_MAX;
        m_max_plain_q_val = Q_VAL_MIN;
    }

    void print() const override {
        printf("### Relu Layer ###\n");
        printf("Input dims: ");
        for (auto dim : m_input_dims) printf("%lu ", dim);
        printf("\n");
        printf("Output dims: ");
        for (auto dim : m_output_dims) printf("%lu ", dim);
        printf("\n");
    }
};

#endif
#ifndef SIGN_H
#define SIGN_H

#include <vector>
#include <algorithm>
#include <cstdlib>

#include "misc/datatypes.h"
#include "layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

template <typename T>
T sign_fun(T x) {
    return x >= 0 ? 1 : -1;
}

class Sign : public Layer {
   public:
    Sign(dim_t input_dims) : Layer(input_dims, input_dims) {}

    LayerType get_type() const override { return LayerType::sign; }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        ScalarTensor<wandb_t> output(input.get_dims());
        for (size_t i = 0; i < input.size(); i++) {
            output.push_back(sign_fun(input[i]));
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
            output.push_back(sign_fun(input[i]));
        }
        if (track_extreme_values) {
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
        printf("### Sign Layer ###\n");
        printf("Input dims: ");
        for (auto dim : m_input_dims) printf("%lu ", dim);
        printf("\n");
        printf("Output dims: ");
        for (auto dim : m_output_dims) printf("%lu ", dim);
        printf("\n");
    }
};

#endif
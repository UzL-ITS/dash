#ifndef DENSE_H
#define DENSE_H

#include <algorithm>
#include <cassert>
#include <vector>

#include "circuit/layer/layer.h"
#include "circuit/scalar_tensor.h"
#include "misc/datatypes.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

using std::vector;

class Dense : public Layer {
    ScalarTensor<wandb_t> m_weights;
    ScalarTensor<wandb_t> m_biases;
    int m_channel_tf;

    // Quantized weights and biases
    ScalarTensor<q_val_t> m_q_weights;
    ScalarTensor<q_val_t> m_q_biases;
    QuantizationMethod m_q_method;
    wandb_t m_q_const;

   public:
    Dense(ScalarTensor<wandb_t> weights, ScalarTensor<wandb_t> biases,
          QuantizationMethod q_method = QuantizationMethod::SimpleQuant,
          wandb_t q_const = 10, int channel_tf = 0)
        : Layer(dim_t{weights.get_dims().at(1)},
                dim_t{weights.get_dims().at(0)}),
          m_weights(weights),
          m_biases(biases),
          m_channel_tf(channel_tf) {
        assert(m_weights.get_dims()[0] == m_biases.get_dims()[0] &&
               "Weights and biases dimensions do not match");

        if (q_method == QuantizationMethod::SimpleQuant) {
            m_q_weights =
                ScalarTensor<q_val_t>::quantize(weights, q_method, q_const);
            m_q_biases =
                ScalarTensor<q_val_t>::quantize(biases, q_method, q_const);
        } else {
            m_q_weights =
                ScalarTensor<q_val_t>::quantize(weights, q_method, QL);
            m_q_biases =
                ScalarTensor<q_val_t>::quantize(biases, q_method, 2 * QL);
        }
        m_q_method = q_method;
        m_q_const = q_const;
        m_min_plain_q_val = std::min(m_q_weights.min(), m_q_biases.min());
        m_max_plain_q_val = std::max(m_q_weights.max(), m_q_biases.max());
    }

    Dense(size_t input_dim, size_t output_dim,
          QuantizationMethod q_method = QuantizationMethod::SimpleQuant,
          wandb_t q_const = 10, int channel_tf = 0)
        : Layer(dim_t{input_dim}, dim_t{output_dim}),
          m_q_method{q_method},
          m_q_const{q_const},
          m_channel_tf{channel_tf} {}

    void quantize(wandb_t q_const) override {
        m_q_const = q_const;
        m_q_weights = ScalarTensor<q_val_t>::quantize(
            m_weights, QuantizationMethod::SimpleQuant, q_const);
        m_q_biases = ScalarTensor<q_val_t>::quantize(
            m_biases, QuantizationMethod::SimpleQuant, q_const);
        m_min_plain_q_val = Q_VAL_MAX;
        m_max_plain_q_val = Q_VAL_MIN;
    }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        ScalarTensor<wandb_t> output;
        if (track_extreme_values) {
            m_min_plain_val = std::min(m_min_plain_val, input.min());
            m_max_plain_val = std::max(m_max_plain_val, input.max());
            if (m_channel_tf == 0) {
                output = m_weights.matvecmul(input, &m_min_plain_val,
                                             &m_max_plain_val);
            } else {
                output = m_weights.matvecmul_tf(input, &m_min_plain_val,
                                                &m_max_plain_val, m_channel_tf);
            }
            output += m_biases;
            m_min_plain_val = std::min(m_min_plain_val, output.min());
            m_max_plain_val = std::max(m_max_plain_val, output.max());
        } else {
            if (m_channel_tf == 0) {
                output = m_weights.matvecmul(input);
            } else {
                output = m_weights.matvecmul_tf(input, m_channel_tf);
            }
            output += m_biases;
        }
        return output;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input, bool track_extreme_value = true) override {
        ScalarTensor<q_val_t> output;
        if (track_extreme_value) {
            m_min_plain_q_val = std::min(m_min_plain_q_val, input.min());
            m_max_plain_q_val = std::max(m_max_plain_q_val, input.max());
            if (m_channel_tf == 0) {
                output = m_q_weights.matvecmul(input, &m_min_plain_q_val,
                                               &m_max_plain_q_val);
            } else {
                output =
                    m_q_weights.matvecmul_tf(input, &m_min_plain_q_val,
                                             &m_max_plain_q_val, m_channel_tf);
            }
            output += m_q_biases;
            m_min_plain_q_val = std::min(m_min_plain_q_val, output.min());
            m_max_plain_q_val = std::max(m_max_plain_q_val, output.max());
        } else {
            if (m_channel_tf == 0) {
                output = m_q_weights.matvecmul(input);
            } else {
                output = m_q_weights.matvecmul_tf(input, m_channel_tf);
            }
            output += m_q_biases;
        }

        return output;
    }

    q_val_t get_min_plain_q_val() const override {
        auto min_q_param = std::min(m_q_weights.min(), m_q_biases.min());
        return std::min(m_min_plain_q_val, min_q_param);
    }

    q_val_t get_max_plain_q_val() const override {
        auto max_q_param = std::max(m_q_weights.max(), m_q_biases.max());
        return std::max(m_max_plain_q_val, max_q_param);
    }

    wandb_t get_min_plain_val() const override {
        auto min_param = std::min(m_weights.min(), m_biases.min());
        return std::min(m_min_plain_val, min_param);
    }

    wandb_t get_max_plain_val() const override {
        auto max_param = std::max(m_weights.max(), m_biases.max());
        return std::max(m_max_plain_val, max_param);
    }

    LayerType get_type() const override { return LayerType::dense; }

    int get_channel_tf() { return m_channel_tf; }

    ScalarTensor<wandb_t>& get_weights() {
        assert(m_weights.size() == m_input_dims.at(0) * m_output_dims.at(0) &&
               "Weights not properly initialized");
        return m_weights;
    }

    ScalarTensor<wandb_t>& get_biases() {
        assert(m_biases.size() == m_output_dims.at(0) &&
               "Biases not properly initialized");
        return m_biases;
    }

    ScalarTensor<q_val_t>& get_q_weights() {
        assert(m_q_weights.size() == m_input_dims.at(0) * m_output_dims.at(0) &&
               "Quantized weights not properly initialized");
        return m_q_weights;
    }

    void set_q_weights(ScalarTensor<q_val_t> q_weights) {
        m_q_weights = q_weights;
    }

    ScalarTensor<q_val_t>& get_q_biases() {
        assert(m_q_biases.size() == m_output_dims.at(0) &&
               "Quantized biases not properly initialized");
        return m_q_biases;
    }

    void set_q_biases(ScalarTensor<q_val_t> q_biases) { m_q_biases = q_biases; }

    wandb_t get_q_const() const override { return m_q_const; }

    void print() const override {
        printf("### Dense Layer ###\n");
        printf("Input size: %lu\n", m_input_size);
        printf("Output size: %lu\n", m_output_size);
        printf("Weight dimensions: ");
        for (auto dim : m_weights.get_dims()) {
            printf("%lu ", dim);
        }
        printf("\n");
        printf("Bias size: %lu\n", m_biases.size());
    }
};

#endif
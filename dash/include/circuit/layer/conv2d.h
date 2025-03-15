#ifndef CONV2D_H
#define CONV2D_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

#include "circuit/scalar_tensor.h"
#include "layer.h"
#include "misc/datatypes.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

using std::vector;

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
 * @param[in]  activation     Activation function used.
 *
 * @return     Pointer to the output of the layer.
 */
template <typename T>
void conv2d_fun(T* output, T* input, T* weights, T* bias, size_t input_width,
                size_t input_height, size_t channel, size_t filter,
                size_t filter_width, size_t filter_height, size_t stride_width,
                size_t stride_height) {
    size_t output_width = (input_width - filter_width) / stride_width + 1;
    size_t output_height = (input_height - filter_height) / stride_height + 1;

    size_t output_size = output_width * output_height;  // sizer per filter
    size_t filter_size = filter_width * filter_height;
    size_t input_size = input_width * input_height;

    // for all filter
    for (size_t v = 0; v < filter; ++v) {
        // move filter along y-axis
        for (size_t l = 0; l < output_height; ++l) {
            // move filter along x-axis
            for (size_t k = 0; k < output_width; ++k) {
                // scalar product over all input-channel...
                for (size_t w = 0; w < channel; ++w) {
                    // ...along filter height and...
                    for (size_t i = 0; i < filter_height; ++i) {
                        // ...width
                        for (size_t j = 0; j < filter_width; ++j) {
                            output[v * output_size + l * output_width + k] +=
                                weights[v * filter_size * channel +
                                        w * filter_size + i * filter_height +
                                        j] *
                                input[w * input_size + i * input_width + j +
                                      k * stride_width +
                                      l * stride_height * input_width];
                        }
                    }
                }
                // add bias to each output-channel and apply activation function
                output[v * output_size + l * output_width + k] += bias[v];
            }
        }
    }
}

template <typename T>
void conv2d_fun(T* output, T* input, T* weights, T* bias, size_t input_width,
                size_t input_height, size_t channel, size_t filter,
                size_t filter_width, size_t filter_height, size_t stride_width,
                size_t stride_height, T* min, T* max) {
    size_t output_width = (input_width - filter_width) / stride_width + 1;
    size_t output_height = (input_height - filter_height) / stride_height + 1;

    size_t output_size = output_width * output_height;  // sizer per filter
    size_t filter_size = filter_width * filter_height;
    size_t input_size = input_width * input_height;

    // for all filter
    for (size_t v = 0; v < filter; ++v) {
        // move filter along y-axis
        for (size_t l = 0; l < output_height; ++l) {
            // move filter along x-axis
            for (size_t k = 0; k < output_width; ++k) {
                // scalar product over all input-channel...
                for (size_t w = 0; w < channel; ++w) {
                    // ...along filter height and...
                    for (size_t i = 0; i < filter_height; ++i) {
                        // ...width
                        for (size_t j = 0; j < filter_width; ++j) {
                            T tmp = weights[v * filter_size * channel +
                                            w * filter_size +
                                            i * filter_height + j] *
                                    input[w * input_size + i * input_width + j +
                                          k * stride_width +
                                          l * stride_height * input_width];

                            *min = std::min(*min, tmp);
                            *max = std::max(*max, tmp);

                            output[v * output_size + l * output_width + k] +=
                                tmp;

                            *min = std::min(
                                *min,
                                output[v * output_size + l * output_width + k]);
                            *max = std::max(
                                *max,
                                output[v * output_size + l * output_width + k]);
                        }
                    }
                }
                // add bias to each output-channel and apply activation function
                output[v * output_size + l * output_width + k] += bias[v];

                T tmp = output[v * output_size + l * output_width + k];
                *min = std::min(*min, tmp);
                *max = std::max(*max, tmp);
            }
        }
    }
}

/**
 * @brief Conv2D
 * @details No support for padding (only 'valid'), dilation and grouped-conv.
 * planned.
 *
 * TensorFlow semantic:
 * Input: number of images x input-height x input-width x channels
 * Filter: filter-height x filter-width x channels x number of filters
 *
 */
class Conv2d : public Layer {
    ScalarTensor<wandb_t> m_weights;
    ScalarTensor<wandb_t> m_biases;

    // Quantized weights and biases
    ScalarTensor<q_val_t> m_q_weights;
    ScalarTensor<q_val_t> m_q_biases;

    size_t m_input_width;
    size_t m_input_height;
    size_t m_channel;
    size_t m_filter;
    size_t m_filter_width;
    size_t m_filter_height;
    size_t m_stride_width;
    size_t m_stride_height;

    QuantizationMethod m_q_method;
    wandb_t m_q_const;

   public:
    Conv2d(ScalarTensor<wandb_t> weights, ScalarTensor<wandb_t> biases,
           size_t input_width, size_t input_height, size_t channel,
           size_t filter, size_t filter_width, size_t filter_height,
           size_t stride_width, size_t stride_height,
           int ql, QuantizationMethod q_method = QuantizationMethod::SimpleQuant,
           wandb_t q_const = 10)
        : Layer(
              dim_t{input_width, input_height, channel},
              dim_t{out_w(input_width, filter_width, stride_width),
                    out_h(input_height, filter_height, stride_height), filter}),
          m_weights(weights),
          m_biases(biases) {
        if (q_method == QuantizationMethod::SimpleQuant) {
            m_q_weights =
                ScalarTensor<q_val_t>::quantize(weights, q_method, q_const);
            m_q_biases =
                ScalarTensor<q_val_t>::quantize(biases, q_method, q_const);
        } else {
            m_q_weights =
                ScalarTensor<q_val_t>::quantize(weights, q_method, ql);
            m_q_biases =
                ScalarTensor<q_val_t>::quantize(biases, q_method, 2 * ql);
        }
        m_input_width = input_width;
        m_input_height = input_height;
        m_channel = channel;
        m_filter = filter;
        m_filter_width = filter_width;
        m_filter_height = filter_height;
        m_stride_width = stride_width;
        m_stride_height = stride_height;
        m_q_method = q_method;
        m_q_const = q_const;
        m_min_plain_q_val = std::min(m_q_weights.min(), m_q_biases.min());
        m_max_plain_q_val = std::max(m_q_weights.max(), m_q_biases.max());
    }

    Conv2d(size_t input_width, size_t input_height, size_t channel,
           size_t filter, size_t filter_width, size_t filter_height,
           size_t stride_width, size_t stride_height,
           QuantizationMethod q_method = QuantizationMethod::SimpleQuant,
           wandb_t q_const = 10)
        : Layer(
              dim_t{input_height, input_width, channel},
              dim_t{out_w(input_width, filter_width, stride_width),
                    out_h(input_height, filter_height, stride_height), filter}),
          m_input_width{input_width},
          m_input_height{input_height},
          m_channel{channel},
          m_filter{filter},
          m_filter_width{filter_width},
          m_filter_height{filter_height},
          m_stride_width{stride_width},
          m_stride_height{stride_height},
          m_q_method{q_method},
          m_q_const{q_const} {}

    void quantize(wandb_t q_const) override {
        m_q_const = q_const;
        m_q_weights = ScalarTensor<q_val_t>::quantize(
            m_weights, QuantizationMethod::SimpleQuant, q_const);
        m_q_biases = ScalarTensor<q_val_t>::quantize(
            m_biases, QuantizationMethod::SimpleQuant, q_const);
        m_min_plain_q_val = Q_VAL_MAX;
        m_max_plain_q_val = Q_VAL_MIN;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override {
        ScalarTensor<q_val_t> output{(q_val_t)0, m_output_dims};
        if (track_extreme_values) {
            m_min_plain_q_val = std::min(m_min_plain_q_val, input.min());
            m_max_plain_q_val = std::max(m_max_plain_q_val, input.max());
            conv2d_fun(output.data(), input.data(), m_q_weights.data(),
                       m_q_biases.data(), m_input_width, m_input_height,
                       m_channel, m_filter, m_filter_width, m_filter_height,
                       m_stride_width, m_stride_height, &m_min_plain_q_val,
                       &m_max_plain_q_val);
        } else {
            conv2d_fun(output.data(), input.data(), m_q_weights.data(),
                       m_q_biases.data(), m_input_width, m_input_height,
                       m_channel, m_filter, m_filter_width, m_filter_height,
                       m_stride_width, m_stride_height);
        }
        return output;
    }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        ScalarTensor<wandb_t> output{(wandb_t)0, m_output_dims};
        if (track_extreme_values) {
            m_min_plain_val = std::min(m_min_plain_val, input.min());
            m_max_plain_val = std::max(m_max_plain_val, input.max());
            conv2d_fun(output.data(), input.data(), m_weights.data(),
                       m_biases.data(), m_input_width, m_input_height,
                       m_channel, m_filter, m_filter_width, m_filter_height,
                       m_stride_width, m_stride_height, &m_min_plain_val,
                       &m_max_plain_val);
        } else {
            conv2d_fun(output.data(), input.data(), m_weights.data(),
                       m_biases.data(), m_input_width, m_input_height,
                       m_channel, m_filter, m_filter_width, m_filter_height,
                       m_stride_width, m_stride_height);
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

    LayerType get_type() const override { return LayerType::conv2d; }

    ScalarTensor<wandb_t>& get_weights() {
        size_t size = m_filter_height * m_filter_width * m_channel * m_filter;
        assert(m_weights.size() == size && "Weights not properly initialized");
        return m_weights;
    }

    ScalarTensor<wandb_t>& get_biases() {
        assert(m_biases.size() == m_filter &&
               "Biases not properly initialized");
        return m_biases;
    }

    ScalarTensor<q_val_t>& get_q_weights() {
        size_t size = m_filter_height * m_filter_width * m_channel * m_filter;
        assert(m_q_weights.size() == size &&
               "Quantized and encoded weights not properly initialized");
        return m_q_weights;
    }

    void set_q_weights(ScalarTensor<q_val_t> q_weights) {
        m_q_weights = q_weights;
    }

    ScalarTensor<q_val_t>& get_q_biases() {
        assert(m_q_biases.size() == m_filter &&
               "Quantized and encoded biases not properly initialized");
        return m_q_biases;
    }

    void set_q_biases(ScalarTensor<q_val_t> q_biases) { m_q_biases = q_biases; }

    int get_input_width() const { return m_input_width; }
    int get_input_height() const { return m_input_height; }
    int get_channel() const { return m_channel; }
    int get_filter() const { return m_filter; }
    int get_filter_width() const { return m_filter_width; }
    int get_filter_height() const { return m_filter_height; }
    int get_stride_width() const { return m_stride_width; }
    int get_stride_height() const { return m_stride_height; }

    wandb_t get_q_const() const override { return m_q_const; }

   private:
    static size_t out_w(size_t input_width, size_t filter_width,
                        size_t stride_width) {
        assert((input_width - filter_width) % stride_width == 0 &&
               "Invalid stride width");
        return (input_width - filter_width) / stride_width + 1;
    }

    static size_t out_h(size_t input_height, size_t filter_height,
                        size_t stride_height) {
        assert((input_height - filter_height) % stride_height == 0 &&
               "Invalid stride height");
        return (input_height - filter_height) / stride_height + 1;
    }

    void print() const override {
        printf("### Conv2D Layer ###\n");
        printf("Input width: %lu\n", m_input_width);
        printf("Input height: %lu\n", m_input_height);
        printf("Channel: %lu\n", m_channel);
        printf("Filter: %lu\n", m_filter);
        printf("Filter width: %lu\n", m_filter_width);
        printf("Filter height: %lu\n", m_filter_height);
        printf("Stride width: %lu\n", m_stride_width);
        printf("Stride height: %lu\n", m_stride_height);
        printf("Output width: %lu\n", m_output_dims[1]);
        printf("Output height: %lu\n", m_output_dims[0]);
        printf("Output channel: %lu\n", m_output_dims[2]);
        printf("Weight dimensions: ");
        for (auto dim : m_weights.get_dims()) printf("%lu ", dim);
        printf("\n");
        printf("Bias size: %ld\n", m_biases.size());
        printf("Input dimensions: ");
        for (auto dim : m_input_dims) printf("%lu ", dim);
        printf("\n");
        printf("Output dimensions: ");
        for (auto dim : m_output_dims) printf("%lu ", dim);
        printf("\n");
    }
};

#endif
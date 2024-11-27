#ifndef LAYER_H
#define LAYER_H

#include <numeric>
#include <vector>
#include <cstdlib>

#include "misc/datatypes.h"
#include "circuit/scalar_tensor.h"

using std::vector;

class Layer {
   protected:
    dim_t m_input_dims;
    dim_t m_output_dims;
    size_t m_input_size;
    size_t m_output_size;

    wandb_t m_min_plain_val{WANDB_VAL_MAX};
    wandb_t m_max_plain_val{WANDB_VAL_MIN};

    q_val_t m_min_plain_q_val{Q_VAL_MAX};
    q_val_t m_max_plain_q_val{Q_VAL_MIN};

   public:
    enum LayerType {
        dense,
        conv2d,
        sum_layer,
        projection,
        mult_layer,
        mixed_mod_mult_layer,
        approx_relu,
        flatten,
        sign,
        max_pool,
        rescale
    };

    Layer(dim_t input_dims, dim_t output_dims)
        : m_input_dims{input_dims}, m_output_dims{output_dims} {
        m_input_size =
            std::accumulate(m_input_dims.begin(), m_input_dims.end(), 1lu,
                            [](size_t a, size_t b) { return a * b; });
        m_output_size =
            std::accumulate(m_output_dims.begin(), m_output_dims.end(), 1lu,
                            [](size_t a, size_t b) { return a * b; });
    }

    virtual ~Layer() {}

    virtual LayerType get_type() const = 0;
    virtual dim_t get_input_dims() const { return m_input_dims; }
    virtual dim_t get_output_dims() const { return m_output_dims; }
    virtual size_t get_input_size() const { return m_input_size; }
    virtual size_t get_output_size() const { return m_output_size; }
    virtual q_val_t get_min_plain_q_val() const { return m_min_plain_q_val; }
    virtual q_val_t get_max_plain_q_val() const{ return m_max_plain_q_val; }
    virtual q_val_t get_range_plain_q_val() const {
        return std::abs(m_max_plain_q_val) + std::abs(m_min_plain_q_val);
    }
    virtual wandb_t get_min_plain_val() const { return m_min_plain_val; }
    virtual wandb_t get_max_plain_val() const { return m_max_plain_val; }
    virtual wandb_t get_range_plain_val() const {
        return std::abs(m_max_plain_val) + std::abs(m_min_plain_val);
    }
    virtual ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input, bool track_extreme_values = true) = 0;
    virtual ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input, bool track_extreme_values = true) = 0;
    virtual void print() const = 0;
    virtual wandb_t get_q_const() const { return 0; };
    virtual void quantize(wandb_t q_const) = 0;
};

#endif
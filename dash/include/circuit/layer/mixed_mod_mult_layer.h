#ifndef MIXED_MOD_MULT_LAYER_H
#define MIXED_MOD_MULT_LAYER_H
// For testing purposes

#include <vector>

#include "misc/datatypes.h"
#include "layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

using std::vector;

class MixedModMultLayer : public Layer {
    crt_val_t m_smaller_modulus;

   public:
    /**
     * @brief Construct a new MixedModMultLayer object
     *
     * @param input_dims
     * @param output_dims
     */
    MixedModMultLayer(dim_t input_dims, dim_t output_dims,
                      crt_val_t smaller_modulus)
        : Layer(input_dims, output_dims), m_smaller_modulus{smaller_modulus} {}

    void quantize(wandb_t q_const) override {}

    // Dummy evaluation functions
    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        return input;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override {
        return input;
    }

    LayerType get_type() const override { return LayerType::mixed_mod_mult_layer; }

    crt_val_t get_smaller_modulus() const { return m_smaller_modulus; }

    void print() const override { printf("### MixedModMult Layer ###\n"); }
};

#endif
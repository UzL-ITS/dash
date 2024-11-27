#ifndef MULT_LAYER_H
#define MULT_LAYER_H
// For testing purposes

#include "misc/datatypes.h"
#include "layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

class MultLayer : public Layer {
   public:
    /**
     * @brief Construct a new ProdLayer object
     *
     * @param input_dims
     * @param output_dims
     */
    MultLayer(dim_t input_dims, dim_t output_dims)
        : Layer(input_dims, output_dims) {}

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

    LayerType get_type() const override { return LayerType::mult_layer; }

    void print() const override{
        printf("### Mult Layer ###\n");
    }
};

#endif
#ifndef PROJECTION_H
#define PROJECTION_H
// For testing purposes

#include <vector>

#include "misc/datatypes.h"
#include "layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

using std::vector;

class Projection : public Layer {
    vector<crt_val_t> m_in_moduli;
    vector<crt_val_t> m_out_moduli;
    crt_val_t (*m_functionality)(crt_val_t, void*);

   public:
    /**
     * @brief Construct a new Projection object
     *
     * Caution: If CRT-Base of Garbled Circuit is greater than 1, only linear
     * functionalities are useful.
     *
     * @param input_dims
     * @param out_moduli
     * @param functionality
     */
    Projection(dim_t input_dims, vector<crt_val_t> in_moduli,
               vector<crt_val_t> out_moduli,
               crt_val_t (*functionality)(crt_val_t, void*))
        : Layer(input_dims, input_dims),
          m_in_moduli{in_moduli},
          m_out_moduli{out_moduli},
          m_functionality{functionality} {};

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

    LayerType get_type() const override { return LayerType::projection; }

    vector<crt_val_t> get_in_moduli() const { return m_in_moduli; }
    vector<crt_val_t> get_out_moduli() const { return m_out_moduli; }
    crt_val_t (*get_functionality())(crt_val_t, void*) {
        return m_functionality;
    }

    void print() const override { printf("### Projection Layer ###\n"); }
};

#endif
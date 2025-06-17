#ifndef BASE_EXTENSION_H
#define BASE_EXTENSION_H

#include <algorithm>

#include "circuit/scalar_tensor.h"
#include "layer.h"
#include "misc/datatypes.h"

/**
 * This is just for testing purposes.
 * cf. GarbledBaseExtension
 * @author Felix Maurer
 */
class BaseExtension : public Layer
{
    const vector<crt_val_t> M_EXTRA_MODULI;

public:
    BaseExtension(dim_t input_dims, vector<crt_val_t> extra_moduli) : Layer(input_dims, input_dims), M_EXTRA_MODULI{extra_moduli} {}

    LayerType get_type() const override { return LayerType::base_extension; }

    vector<crt_val_t> get_extra_moduli() const { return M_EXTRA_MODULI; }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override
    {
        return input;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override
    {
        return input;
    }

    void quantize(wandb_t q_const) override
    {
        // No changes needed
    }

    void print() const override
    {
        printf("### Base Extension Layer ###\n");
        printf("Input dims: ");
        for (auto dim : m_input_dims)
            printf("%lu ", dim);
        printf("\n");
        printf("Output dims: ");
        for (auto dim : m_output_dims)
            printf("%lu ", dim);
        printf("\n");
    }
};

#endif
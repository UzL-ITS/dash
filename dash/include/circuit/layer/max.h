#ifndef MAX_H
#define MAX_H

#include <algorithm>
#include <cstdlib>

#include "circuit/scalar_tensor.h"
#include "layer.h"
#include "misc/datatypes.h"

template <typename T>
T GENERIC_MAX(T x, T y)
{
    return x > y ? x : y;
}

/**
 * This is just for testing purposes.
 * @author Felix Maurer
 */
class Max : public Layer
{
public:
    Max() : Layer({2}, {1}) {}

    LayerType get_type() const override { return LayerType::max; }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override
    {
        assert(input.size() == 2);

        ScalarTensor<wandb_t> output({1});
        output.push_back(GENERIC_MAX(input[0], input[1]));

        return output;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override
    {
        assert(input.size() == 2);

        ScalarTensor<q_val_t> output({1});
        output.push_back(GENERIC_MAX(input[0], input[1]));

        return output;
    }

    void quantize(wandb_t q_const) override
    {
        // No changes needed
    }

    void print() const override
    {
        printf("### Max Layer ###\n");
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
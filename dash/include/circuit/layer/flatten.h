#ifndef FLATTEN_H
#define FLATTEN_H

#include <cstdio>
#include <numeric>
#include <cstdlib>

#include "misc/datatypes.h"
#include "layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

class Flatten : public Layer {
   public:
    Flatten(dim_t input_dims)
        : Layer(input_dims, dim_t{std::accumulate(
                                begin(input_dims), end(input_dims), 1lu,
                                [](size_t a, size_t b) { return a * b; })}) {}

    virtual ~Flatten() {}

    LayerType get_type() const override { return LayerType::flatten; }

    void quantize(wandb_t q_const) override {}

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        auto output{input};
        output.set_dims(dim_t{m_output_size});
        if (track_extreme_values) {
            m_max_plain_q_val = 0;
            m_min_plain_q_val = 0;
        }
        return output;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override {
        auto output{input};
        output.set_dims(dim_t{m_output_size});
        if (track_extreme_values) {
            m_max_plain_q_val = 0;
            m_min_plain_q_val = 0;
        }
        return output;
    }

    void print() const override {
        printf("### Flatten Layer ###\n");
        printf("Input dims: ");
        for (auto dim : m_input_dims) printf("%lu ", dim);
        printf("\n");
        printf("Output size: %lu\n", m_output_size);
    }
};

#endif
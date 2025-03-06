#ifndef RESCALE_H
#define RESCALE_H

#include "misc/datatypes.h"
#include "circuit/layer/layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

class Rescale : public Layer
{
    int m_l;
    vector<crt_val_t> m_s;

    bool m_use_sign_base_extension;

public:
    Rescale(int l, dim_t dims) : Layer(dims, dims), m_l{l}, m_s{-1}, m_use_sign_base_extension{true} {}
    Rescale(vector<crt_val_t> s, dim_t dims) : Layer(dims, dims), m_l{-1}, m_s{s}, m_use_sign_base_extension{false} {}

    LayerType get_type() const override { return LayerType::rescale; }

    void quantize(wandb_t q_const) override {}

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override
    {
        auto output{input};
        if (track_extreme_values)
        {
            m_max_plain_q_val = 0;
            m_min_plain_q_val = 0;
        }
        return output;
    }

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override
    {
        ScalarTensor<q_val_t> output;
        if (m_use_sign_base_extension)
        {
            output = ScalarTensor<q_val_t>::rescale(input, m_l);
        }
        else
        {
            output = input;
            for (const auto factor : m_s)
            {
                output = ScalarTensor<q_val_t>::rescale(output, factor);
            }
        }
        if (track_extreme_values)
        {
            auto max = std::max(output.max(), input.max());
            m_max_plain_q_val = std::max(m_max_plain_q_val, max);
            auto min = std::min(output.min(), input.min());
            m_min_plain_q_val = std::min(m_min_plain_q_val, min);
        }
        return output;
    }

    void print() const override
    {
        printf("### Rescale Layer ###\n");
        printf("Input dims: ");
        for (auto dim : m_input_dims)
            printf("%lu ", dim);
        printf("\n");
        printf("Output size: %lu\n", m_output_size);
        printf("l: %d\n", m_l);
        printf("s: ");
        for (auto factor : m_s)
            printf("%d ", factor);
    }

    int get_l() const { return m_l; }

    vector<crt_val_t> get_s() const { return m_s; }

    bool get_use_sign_base_extension() const { return m_use_sign_base_extension; }
};

#endif // RESCALE_H
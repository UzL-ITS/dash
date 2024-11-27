#ifndef MAX_POOL2D_H
#define MAX_POOL2D_H

#include <cstdlib>

#include "misc/datatypes.h"
#include "circuit/layer/layer.h"
#include "circuit/scalar_tensor.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

class MaxPool2d : public Layer {
    size_t m_input_width;
    size_t m_input_height;
    size_t m_channel;
    size_t m_kernel_height;
    size_t m_kernel_width;
    size_t m_stride_height;
    size_t m_stride_width;

   public:
    MaxPool2d(size_t input_width, size_t input_height, size_t channel,
              size_t kernel_width, size_t kernel_height,
              size_t stride_width = 1, size_t stride_height = 1)
        : Layer(dim_t{input_width, input_height, channel},
                dim_t{input_width / stride_width, input_height / stride_height,
                      channel}),
          m_input_width{input_width},
          m_input_height{input_height},
          m_channel{channel},
          m_kernel_height{kernel_height},
          m_kernel_width{kernel_width},
          m_stride_height{stride_height},
          m_stride_width{stride_width} {}

    LayerType get_type() const override { return LayerType::max_pool; }

    void quantize(wandb_t q_const) override {}

    ScalarTensor<q_val_t> plain_q_eval(
        ScalarTensor<q_val_t> input,
        bool track_extreme_values = true) override {
        auto output = input.max_pool(m_kernel_width, m_kernel_height,
                                     m_stride_width, m_stride_height);

        if (track_extreme_values) {
            m_min_plain_q_val = output.min();
            m_max_plain_q_val = output.max();
        }

        return output;
    }

    ScalarTensor<wandb_t> plain_eval(
        ScalarTensor<wandb_t> input,
        bool track_extreme_values = true) override {
        auto output = input.max_pool(m_kernel_width, m_kernel_height,
                                     m_stride_width, m_stride_height);

        if (track_extreme_values) {
            m_min_plain_q_val = output.min();
            m_max_plain_q_val = output.max();
        }

        return output;
    }

    void print() const override {
        printf("### MaxPool Layer ###\n");
        printf("Input width: %lu\n", m_input_width);
        printf("Input height: %lu\n", m_input_height);
        printf("Channel: %lu\n", m_channel);
        printf("Kernel width: %lu\n", m_kernel_width);
        printf("Kernel height: %lu\n", m_kernel_height);
        printf("Stride width: %lu\n", m_stride_width);
        printf("Stride height: %lu\n", m_stride_height);
        printf("Input dimensions: ");
        for (auto dim : m_input_dims) {
            printf("%lu ", dim);
        }
        printf("\n");
        printf("Output dimensions: ");
        for (auto dim : m_output_dims) {
            printf("%lu ", dim);
        }
        printf("\n");
    }
};

#endif /* MAX_POOL2D_H */
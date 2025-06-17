#ifndef GARBLED_MAXPOOL2D_H
#define GARBLED_MAXPOOL2D_H

// #include <cstdlib>
#include <vector>
#include <map>

#include "garbling/garbled_circuit_interface.h"
#include "garbling/layer/garbled_layer.h"
#include "garbling/gadgets/max_gadget.h"
#include "circuit/layer/max_pool2d.h"

using std::vector;
using std::map;

/**
 * @class GarbledMaxPool2d
 * @brief Represents a garbled max pooling layer in a neural network.
 * @author Felix Maurer
 *
 * The GarbledMaxPool2d class is responsible for garbling and evaluating a max pooling layer
 * using the garbled circuit technique. It inherits from the GarbledLayer class and implements
 * the garble() and cpu_evaluate() methods. The class also provides methods for CUDA evaluation,
 * but they are not yet implemented.
 *
 * The max pooling operation reduces the spatial dimensions of the input tensor by taking the
 * maximum value within a sliding window. This layer is commonly used in convolutional neural
 * networks to downsample the feature maps and extract the most important features.
 *
 * The garble() method garbles the max pooling layer by performing the necessary computations
 * on the input labels. The cpu_evaluate() method evaluates the garbled max pooling layer on
 * the CPU by performing the same computations as the garble() method, but using encoded inputs.
 *
 * Note: The CUDA evaluation methods are not yet implemented and will throw an exception if called.
 */

class GarbledMaxPool2d : public GarbledLayer
{
public:
    GarbledMaxPool2d(Layer *layer_ptr, GarbledCircuitInterface *gc)
        : GarbledLayer{layer_ptr, gc},
          M_NR_INPUTS{m_layer->get_input_size()},
          M_OUTPUT_DIMS{m_layer->get_output_dims()},
          M_KERNEL_WIDTH{((MaxPool2d *)layer_ptr)->get_kernel_width()},
          M_KERNEL_HEIGHT{((MaxPool2d *)layer_ptr)->get_kernel_height()},
          M_STRIDE_WIDTH{((MaxPool2d *)layer_ptr)->get_stride_width()},
          M_STRIDE_HEIGHT{((MaxPool2d *)layer_ptr)->get_stride_height()}
    {
    }

    ~GarbledMaxPool2d()
    {
        // Clean up MaxGadget dictionary
        for (auto &max_gadget : m_max_gadgets)
        {
            delete max_gadget.second;
        }
        m_max_gadgets.clear();

        // Clean up output labels
        free_out_label();
    }

    void garble(vector<LabelTensor *> *in_label) override
    {
        // Dims are identical for all vector entries, therefore read from .at(0)
        const size_t input_width = m_gc->get_base_label().at(0)->get_dims().at(0);
        const size_t input_height = m_gc->get_base_label().at(0)->get_dims().at(1);
        const size_t input_channel = m_gc->get_base_label().at(0)->get_dims().at(2);
        const size_t output_width = (input_width - M_KERNEL_WIDTH) / M_STRIDE_WIDTH + 1;
        const size_t output_height = (input_height - M_KERNEL_HEIGHT) / M_STRIDE_HEIGHT + 1;
        const size_t output_channel = input_channel;
        const vector<crt_val_t> crt_base = m_gc->get_crt_base();
        const size_t crt_base_size = crt_base.size();

        // Reserve output label
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            m_out_label->push_back(new LabelTensor{crt_base.at(crt_idx), M_OUTPUT_DIMS});
        }

#ifndef SGX
//#pragma omp parallel for collapse(2) //Note: Currently disabled, because MaxGadget allocation is not thread-safe
#endif
        for (size_t i = 0; i < output_width; ++i)
        {
            for (size_t j = 0; j < output_height; ++j)
            {
                for (size_t k = 0; k < output_channel; ++k)
                {
                    // Get vector of LTs containing only the current local max
                    auto cur_max = vector<LabelTensor *>{};
                    for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
                    {
                        LabelSlice slice = in_label->at(crt_idx)->slice({{i * M_STRIDE_WIDTH, i * M_STRIDE_WIDTH}, {j * M_STRIDE_HEIGHT, j * M_STRIDE_HEIGHT}, {k, k}});
                        cur_max.push_back(new LabelTensor(slice, dim_t{1}));
                    }

                    for (size_t l = 0; l < M_KERNEL_WIDTH; ++l)
                    {
                        for (size_t m = 0; m < M_KERNEL_HEIGHT; ++m)
                        {
                            const size_t x = i * M_STRIDE_WIDTH + l;
                            const size_t y = j * M_STRIDE_HEIGHT + m;
                            const size_t z = k;

                            // Get vector of LTs containing the current comparison value
                            auto comp = vector<LabelTensor *>{};
                            for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
                            {
                                LabelSlice slice = in_label->at(crt_idx)->slice({{x, x}, {y, y}, {z, z}});
                                comp.push_back(new LabelTensor(slice, dim_t{1}));
                            }

                            m_max_gadgets[{i, j, k, l, m}] = new MaxGadget(m_gc, M_NR_INPUTS, {2});
                            m_max_gadgets[{i, j, k, l, m}]->garble(&cur_max, &comp, &cur_max);

                            // Clean up
                            for (auto label : comp)
                            {
                                delete label;
                            }
                        }
                    }

                    for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
                    {
                        // Write max of kernel into output label
                        m_out_label->at(crt_idx)->set_label(cur_max.at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX),
                                                            {i, j, k});
                    }

                    // Clean up
                    for (auto label : cur_max)
                    {
                        delete label;
                    }
                }
            }
        }
    }

    vector<LabelTensor *> *cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                                        int nr_threads) override
    {
        free_out_label();

        // Dims are identical for all vector entries, therefore read from .at(0)
        const size_t input_width = m_gc->get_base_label().at(0)->get_dims().at(0);
        const size_t input_height = m_gc->get_base_label().at(0)->get_dims().at(1);
        const size_t input_channel = m_gc->get_base_label().at(0)->get_dims().at(2);
        const size_t output_width = (input_width - M_KERNEL_WIDTH) / M_STRIDE_WIDTH + 1;
        const size_t output_height = (input_height - M_KERNEL_HEIGHT) / M_STRIDE_HEIGHT + 1;
        const size_t output_channel = input_channel;
        const vector<crt_val_t> crt_base = m_gc->get_crt_base();
        const size_t crt_base_size = crt_base.size();

        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            // Reserve output label
            m_out_label->push_back(new LabelTensor{crt_base.at(crt_idx), M_OUTPUT_DIMS});
        }

#ifndef SGX
#pragma omp parallel for num_threads(nr_threads) collapse(2)
#endif
        for (size_t i = 0; i < output_width; ++i)
        {
            for (size_t j = 0; j < output_height; ++j)
            {
                for (size_t k = 0; k < output_channel; ++k)
                {
                    // Get vector of LTs containing only the current local max
                    auto cur_max = vector<LabelTensor *>{};
                    for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
                    {
                        LabelSlice slice = encoded_inputs->at(crt_idx)->slice({{i * M_STRIDE_WIDTH, i * M_STRIDE_WIDTH}, {j * M_STRIDE_HEIGHT, j * M_STRIDE_HEIGHT}, {k, k}});
                        cur_max.push_back(new LabelTensor(slice));
                    }

                    for (size_t l = 0; l < M_KERNEL_WIDTH; ++l)
                    {
                        for (size_t m = 0; m < M_KERNEL_HEIGHT; ++m)
                        {
                            const size_t x = i * M_STRIDE_WIDTH + l;
                            const size_t y = j * M_STRIDE_HEIGHT + m;
                            const size_t z = k;

                            // Get vector of LTs containing the current comparison value
                            auto comp = vector<LabelTensor *>{};
                            for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
                            {
                                LabelSlice slice = encoded_inputs->at(crt_idx)->slice({{x, x}, {y, y}, {z, z}});
                                comp.push_back(new LabelTensor(slice));
                            }

                            m_max_gadgets[{i, j, k, l, m}]->cpu_evaluate(&cur_max, &comp, &cur_max, M_NR_INPUTS);

                            // Clean up
                            for (auto label : comp)
                            {
                                delete label;
                            }
                        }
                    }

                    for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
                    {
                        // Write max of kernel into output label
                        m_out_label->at(crt_idx)->set_label(cur_max.at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX),
                                                            {i, j, k});
                    }

                    // Clean up
                    for (auto label : cur_max)
                    {
                        delete label;
                    }
                }
            }
        }

        return m_out_label;
    }

    void cuda_move() override
    {
        throw "Not yet implemented!";
    }

    void cuda_evaluate(crt_val_t **dev_in_label) override
    {
        throw "Not yet implemented!";
    }

    void cuda_move_output() override
    {
        throw "Not yet implemented!";
    }

    crt_val_t **get_dev_out_label() override
    {
        throw "Not yet implemented!";
    }

private:
    const size_t M_NR_INPUTS;
    const dim_t M_OUTPUT_DIMS;

    const size_t M_KERNEL_WIDTH;
    const size_t M_KERNEL_HEIGHT;
    const size_t M_STRIDE_WIDTH;
    const size_t M_STRIDE_HEIGHT;

    /**
     * Note: This does *not* use elements of dim_t as key, as the vector contains 2*dims-1 entries (i.e. i,j,k,l,m)
    */
    map<vector<size_t>, MaxGadget *> m_max_gadgets;

    /**
     * Use this index to set label of 1 x 1 ... x 1 LabelTensors
     */
    static constexpr size_t SINGLE_ENTRY_TENSOR_DEFAULT_INDEX = 0;
};

#endif
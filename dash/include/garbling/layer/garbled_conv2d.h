#ifndef GARBLED_CONV2D_H
#define GARBLED_CONV2D_H

#include <cstdlib>
#include <cstring>
#include <vector>

#include "circuit/layer/conv2d.h"
#include "circuit/layer/layer.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/label_tensor.h"
#include "garbling/layer/garbled_layer.h"
#include "misc/datatypes.h"
#include "misc/util.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "misc/cuda_util.h"
#endif

using std::vector;

class GarbledConv2d : public GarbledLayer {
    // Quantized and encoded weights and biases
    ScalarTensor<q_val_t> m_qe_weights;
    ScalarTensor<q_val_t> m_qe_biases;
    // Biases get added to output of conv, therefore we need label-type
    vector<LabelTensor*>* m_qe_bias_label{nullptr};
    Conv2d* m_conv;

    crt_val_t** m_dev_out_label{nullptr};
    q_val_t* m_dev_qe_weights{nullptr};
    crt_val_t** m_dev_qe_bias_label{nullptr};

   public:
    GarbledConv2d(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc}, m_conv{static_cast<Conv2d*>(m_layer)} {}

    virtual ~GarbledConv2d() {
#ifndef SGX
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                cudaCheckError(cudaFree(m_dev_out_label[i]));
            }
            delete[] m_dev_out_label;
        }

        if (m_dev_qe_bias_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                cudaCheckError(cudaFree(m_dev_qe_bias_label[i]));
            }
            delete[] m_dev_qe_bias_label;
        }

        if (m_dev_qe_weights != nullptr) {
            cudaCheckError(cudaFree(m_dev_qe_weights));
        }
#else
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                ocall_cudaFree(m_dev_out_label[i]);
            }
            ocall_free(m_dev_out_label);
        }

        if (m_dev_qe_bias_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                ocall_cudaFree(m_dev_qe_bias_label[i]);
            }
            ocall_free(m_dev_qe_bias_label);
        }

        if (m_dev_qe_weights != nullptr) {
            ocall_cudaFree(m_dev_qe_weights);
        }
#endif

        if (m_qe_bias_label != nullptr) {
            for (auto& l : *m_qe_bias_label) {
                delete l;
            }
            delete m_qe_bias_label;
        }
    }

    void garble(vector<LabelTensor*>* in_label) override {
        m_qe_weights = m_conv->get_q_weights();
        m_qe_weights.mod(m_gc->get_crt_modulus());
        m_qe_biases = m_conv->get_q_biases();
        m_qe_biases.mod(m_gc->get_crt_modulus());

        m_qe_bias_label = m_gc->garble_values(m_qe_biases);

        m_out_label->resize(m_gc->get_crt_base().size());

        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i) {
            auto l = *in_label->at(i);
            crt_val_t modulus{l.get_modulus()};
            auto zero_label = m_gc->get_zero_label(modulus);

            m_out_label->at(i) = LabelTensor::conv2d_static_bias_label_zero(
                l, zero_label, m_qe_weights, m_gc->get_zero_label(modulus),
                m_conv->get_input_width(), m_conv->get_input_height(),
                m_conv->get_channel(), m_conv->get_filter(),
                m_conv->get_filter_width(), m_conv->get_filter_height(),
                m_conv->get_stride_width(), m_conv->get_stride_height());
        }
    }

    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads) override {
        free_out_label();
        m_out_label->resize(m_gc->get_crt_base().size());
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i) {
            auto l = *encoded_inputs->at(i);
            crt_val_t modulus{l.get_modulus()};
            auto zero_label = m_gc->get_zero_label(modulus);

            m_out_label->at(i) = LabelTensor::conv2d_zero(
                l, zero_label, m_qe_weights, *m_qe_bias_label->at(i),
                m_conv->get_input_width(), m_conv->get_input_height(),
                m_conv->get_channel(), m_conv->get_filter(),
                m_conv->get_filter_width(), m_conv->get_filter_height(),
                m_conv->get_stride_width(), m_conv->get_stride_height(),
                nr_threads);
        }

        return m_out_label;
    }

#ifndef SGX

    void cuda_move() override {
        // Allocate memory for the output labels
        // - Allocate pointer array
        m_dev_out_label = new crt_val_t*[m_gc->get_crt_base().size()];

        // - Allocate device array
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t size = m_layer->get_output_size();
            size *= nr_comps * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_out_label[i]), size));
        }

        // Move weights
        size_t size = m_qe_weights.size() * sizeof(q_val_t);
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_qe_weights), size));
        cudaCheckError(cudaMemcpy(m_dev_qe_weights, m_qe_weights.data(), size,
                                  cudaMemcpyHostToDevice));

        // Move bias label
        m_dev_qe_bias_label = new crt_val_t*[m_gc->get_crt_base().size()];
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            size_t size = m_qe_bias_label->at(i)->size() * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_qe_bias_label[i]), size));

            cudaCheckError(cudaMemcpy(m_dev_qe_bias_label[i],
                                      m_qe_bias_label->at(i)->get_components(),
                                      size, cudaMemcpyHostToDevice));
        }
    }

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();

        int nr_streams = m_gc->get_crt_base().size();
        cudaStream_t stream[nr_streams];

        for (int i = 0; i < nr_streams; ++i) {
            cudaCheckError(cudaStreamCreate(&stream[i]));

            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t output_width = m_conv->get_output_dims().at(0);
            size_t output_height = m_conv->get_output_dims().at(1);
            crt_val_t* dev_zero_label = m_gc->get_dev_zero_label(modulus);

            int block_x_dim = nr_comps;
            int block_y_dim =
                std::min(static_cast<int>(sqrt(1024 / nr_comps)), 2);  // = 2;
            int block_z_dim =
                std::min(static_cast<int>(sqrt(1024 / nr_comps)), 4);  // = 2;
            dim3 block(block_x_dim, block_y_dim, block_z_dim);

            // more blocks than sms, so no sms idle
            // - NVIDIA A100-PCIE-40GB
            //   * 108 SMs with 64 cores each
            // - NVIDIA GeForce RTX 3090
            //   * 82 SMs with 128 cores each
            // - NVIDIA GeForce RTX 3060 Ti
            //   * 38 SMs with 128 cores each
            int grid_x_dim = (nr_comps + block_x_dim - 1) / block_x_dim;
            int grid_y_dim = (output_width + block_y_dim - 1) / block_y_dim;
            int grid_z_dim = (output_height + block_z_dim - 1) / block_z_dim;
            dim3 grid(grid_x_dim, grid_y_dim, grid_z_dim);

            Conv2dModZero<<<grid, block>>>(
                m_dev_out_label[i], dev_in_label[i], m_dev_qe_weights,
                m_dev_qe_bias_label[i], dev_zero_label,
                m_conv->get_input_width(), m_conv->get_input_height(), m_conv->get_channel(),
                m_conv->get_filter(), m_conv->get_filter_width(), m_conv->get_filter_height(),
                m_conv->get_stride_width(), m_conv->get_stride_height(),
                nr_comps, modulus);

            cudaCheckError(cudaStreamDestroy(stream[i]));
        }
        cudaDeviceSynchronize();
    }

    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());

        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            size_t nr_comps =
                LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
            int modulus = m_gc->get_crt_base().at(i);
            size_t size = static_cast<size_t>(m_layer->get_output_size());
            size *= nr_comps * sizeof(crt_val_t);
            auto output_dims = m_layer->get_output_dims();
            m_out_label->at(i) = new LabelTensor(modulus, output_dims);
            cudaCheckError(cudaMemcpy(m_out_label->at(i)->get_components(),
                                      m_dev_out_label[i], size,
                                      cudaMemcpyDeviceToHost));
        }
    }

#else

    void cuda_move() override {
        // Allocate memory for the output labels
        // - Allocate pointer array
        int size = m_gc->get_crt_base().size() * sizeof(crt_val_t*);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_out_label),
                              size);

        // - Allocate device array
        size_t output_size = m_layer->get_output_size();
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            size_t nr_comps =
                LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
            size_t size = output_size * nr_comps * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_out_label[i]),
                             size);
        }

        // Move weights
        size = m_qe_weights.size() * sizeof(q_val_t);
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_qe_weights), size);
        sgx_cudaMemcpyToDevice(m_dev_qe_weights, m_qe_weights.data(), size);

        // Move bias labels
        // m_dev_qe_bias_label = new crt_val_t*[m_gc->get_crt_base().size()];
        size = m_gc->get_crt_base().size() * sizeof(crt_val_t*);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_qe_bias_label),
                              size);
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            size_t size = m_qe_bias_label->at(i)->size() * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_qe_bias_label[i]),
                             size);

            sgx_cudaMemcpyToDevice(m_dev_qe_bias_label[i],
                                   m_qe_bias_label->at(i)->get_components(),
                                   size);
        }
    }

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();

        int crt_base_size = m_gc->get_crt_base().size();
        crt_val_t* crt_base = m_gc->get_crt_base().data();
        size_t output_width = m_conv->get_output_dims().at(0);
        size_t output_height = m_conv->get_output_dims().at(1);
        crt_val_t** dev_zero_label = m_gc->get_dev_zero_label();

        ocall_cuda_eval_conv2d(
            m_dev_out_label, dev_in_label, m_dev_qe_weights,
            m_dev_qe_bias_label, dev_zero_label, crt_base_size, crt_base, output_width,
            output_height, m_conv->get_input_width(),
            m_conv->get_input_height(), m_conv->get_channel(),
            m_conv->get_filter(), m_conv->get_filter_width(),
            m_conv->get_filter_height(), m_conv->get_stride_width(),
            m_conv->get_stride_height());
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            auto modulus = m_gc->get_crt_base().at(i);
            auto output_dims = m_layer->get_output_dims();
            m_out_label->at(i) = new LabelTensor(modulus, output_dims);
            crt_val_t* comps = m_out_label->at(i)->get_components();
            size_t size = m_out_label->at(i)->size();
            crt_val_t* tmp;
            ocall_cudaMemcpyFromDevice(reinterpret_cast<void**>(&tmp),
                                       m_dev_out_label[i],
                                       size * sizeof(crt_val_t));
            std::memcpy(comps, tmp, size * sizeof(crt_val_t));
            ocall_free(reinterpret_cast<void*>(tmp));
        }
    }

#endif
    crt_val_t** get_dev_out_label() override { return m_dev_out_label; }
    q_val_t* get_dev_qe_weights() { return m_dev_qe_weights; }
    crt_val_t** get_dev_qe_bias_label() { return m_dev_qe_bias_label; }

    vector<LabelTensor*>* get_qe_bias_label() { return m_qe_bias_label; }
    void set_qe_bias_label(vector<LabelTensor*>* qe_bias_label) {
        m_qe_bias_label = qe_bias_label;
    }
};

#endif
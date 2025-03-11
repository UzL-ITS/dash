#ifndef GARBLED_DENSE_H
#define GARBLED_DENSE_H

#include <cstdlib>
#include <cstring>
#include <vector>

#include "circuit/layer/dense.h"
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

class GarbledDense : public GarbledLayer {
    // Quantized and encoded weights and biases
    ScalarTensor<q_val_t> m_qe_weights;
    ScalarTensor<q_val_t> m_qe_biases;
    // Biases get added to output of gemm, therefore we need label-type
    vector<LabelTensor*>* m_qe_bias_label{nullptr};
    Dense* m_dense{nullptr};

    crt_val_t** m_dev_out_label{nullptr};
    q_val_t* m_dev_qe_weights{nullptr};
    crt_val_t** m_dev_qe_bias_label{nullptr};

   public:
    GarbledDense(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc}, m_dense{static_cast<Dense*>(m_layer)} {}

    virtual ~GarbledDense() {
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
        int channel_tf = m_dense->get_channel_tf();

        // Encode quantized weights and biases
        m_qe_weights = m_dense->get_q_weights();
        m_qe_weights.mod(m_gc->get_crt_modulus());

        m_qe_biases = m_dense->get_q_biases();
        m_qe_biases.mod(m_gc->get_crt_modulus());

        m_qe_bias_label = m_gc->garble_values(m_qe_biases);

        size_t crt_base_size = m_gc->get_crt_base().size();
        m_out_label->resize(crt_base_size);

        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t modulus{m_gc->get_crt_base().at(i)};
            auto in = *in_label->at(i);
            auto zero_label = m_gc->get_zero_label(modulus);
            if (channel_tf == 0) {
#ifdef LABEL_TENSOR_USE_EIGEN
                m_out_label->at(i) = LabelTensor::matvecmul_eigen(m_qe_weights, in, zero_label);
#else
                m_out_label->at(i) = LabelTensor::matvecmul(m_qe_weights, in, zero_label, DEFAULT_NUM_THREADS);
#endif
            } else {
                m_out_label->at(i) = LabelTensor::matvecmul_tf(m_qe_weights, in, zero_label, channel_tf, DEFAULT_NUM_THREADS);
            }
            *m_out_label->at(i) += m_gc->get_zero_label(modulus);
        }
    }

    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads) override {
        free_out_label();

        size_t crt_base_size = m_gc->get_crt_base().size();
        m_out_label->resize(crt_base_size);
        int channel_tf = m_dense->get_channel_tf();

        for (size_t i = 0; i < crt_base_size; ++i) {
            auto in = *encoded_inputs->at(i);
            crt_val_t modulus{m_gc->get_crt_base().at(i)};
            auto zero_label = m_gc->get_zero_label(modulus);
            if (channel_tf == 0) {
#ifdef LABEL_TENSOR_USE_EIGEN
                m_out_label->at(i) = LabelTensor::matvecmul_eigen(m_qe_weights, in, zero_label);
#else
                m_out_label->at(i) = LabelTensor::matvecmul(m_qe_weights, in, zero_label, nr_threads);
#endif
            } else {
                m_out_label->at(i) = LabelTensor::matvecmul_tf(m_qe_weights, in, zero_label, channel_tf, nr_threads);
            }
            *m_out_label->at(i) += *m_qe_bias_label->at(i);
        }
        return m_out_label;
    }
#ifndef SGX
    void cuda_move() override {
        // Allocate memory for the output labels
        // - Allocate pointer array
        m_dev_out_label = new crt_val_t*[m_gc->get_crt_base().size()];
        // - Allocate device array
        size_t output_size = m_layer->get_output_size();
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t size = output_size * nr_comps * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_out_label[i]), size));
        }

        // Move weights
        cudaCheckError(cudaMalloc(reinterpret_cast<void**>(&m_dev_qe_weights),
                                  m_qe_weights.size() * sizeof(q_val_t)));
        cudaCheckError(cudaMemcpy(m_dev_qe_weights, m_qe_weights.data(),
                                  m_qe_weights.size() * sizeof(q_val_t),
                                  cudaMemcpyHostToDevice));

        // Move bias labels
        m_dev_qe_bias_label = new crt_val_t*[m_gc->get_crt_base().size()];
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            cudaCheckError(
                cudaMalloc(reinterpret_cast<void**>(&m_dev_qe_bias_label[i]),
                           m_qe_bias_label->at(i)->size() * sizeof(crt_val_t)));

            cudaCheckError(
                cudaMemcpy(m_dev_qe_bias_label[i],
                           m_qe_bias_label->at(i)->get_components(),
                           m_qe_bias_label->at(i)->size() * sizeof(crt_val_t),
                           cudaMemcpyHostToDevice));
        }
    }

    /**
     * @brief Evaluate the layer on the GPU
     *
     * Perform MatMul (MxK)*(KxN)=(MxN)
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();

        int channel_tf = m_dense->get_channel_tf();

        int nr_streams = m_gc->get_crt_base().size();
        cudaStream_t stream[nr_streams];

        size_t M = m_layer->get_output_dims().at(0);
        size_t K = m_layer->get_input_dims().at(0);

        for (int i = 0; i < nr_streams; ++i) {
            cudaCheckError(cudaStreamCreate(&stream[i]));

            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t N = LabelTensor::get_nr_comps(modulus);
            crt_val_t* dev_zero_label = m_gc->get_dev_zero_label(modulus);

            dim3 block(32, 32);
            dim3 grid((N + block.x) / block.x, (M + block.y) / block.y);
            if (channel_tf == 0) {
                MatMulModAddZero<<<grid, block, 0, stream[i]>>>(
                    m_dev_qe_weights, dev_in_label[i], m_dev_out_label[i],
                    m_dev_qe_bias_label[i], dev_zero_label, modulus, M, N, K);
            } else {
                MatMulModAddZeroTF<<<grid, block, 0, stream[i]>>>(
                    m_dev_qe_weights, dev_in_label[i], m_dev_out_label[i],
                    m_dev_qe_bias_label[i], dev_zero_label, modulus, M, N, K,
                    channel_tf);
            }

            cudaCheckError(cudaStreamDestroy(stream[i]));
        }
        cudaDeviceSynchronize();
    }

    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());

        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            auto crt_base = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(crt_base);
            size_t output_dim = m_layer->get_output_dims().at(0);
            dim_t dims{output_dim};
            m_out_label->at(i) = new LabelTensor(crt_base, dims);
            cudaCheckError(cudaMemcpy(m_out_label->at(i)->get_components(),
                                      m_dev_out_label[i],
                                      output_dim * nr_comps * sizeof(crt_val_t),
                                      cudaMemcpyDeviceToHost));
        }
    }

#else
    /**
     * @brief Allocate memory for output labels on GPU.
     *
     */
    void cuda_move() override {
        // Allocate memory for the output labels
        // - Allocate pointer array
        // m_dev_out_label = new crt_val_t*[m_gc->get_crt_base().size()];
        int size = m_gc->get_crt_base().size() * sizeof(crt_val_t*);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_out_label),
                              size);

        // - Allocate device array
        size_t output_size = m_layer->get_output_size();
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t size = output_size * nr_comps * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_out_label[i]),
                             size);
        }

        // Move weights
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_qe_weights),
                         m_qe_weights.size() * sizeof(q_val_t));
        sgx_cudaMemcpyToDevice(m_dev_qe_weights, m_qe_weights.data(),
                               m_qe_weights.size() * sizeof(q_val_t));

        // Move bias labels
        // m_dev_qe_bias_label = new crt_val_t*[m_gc->get_crt_base().size()];
        size = m_gc->get_crt_base().size() * sizeof(crt_val_t*);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_qe_bias_label),
                              size);
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            ocall_cudaMalloc(
                reinterpret_cast<void**>(&m_dev_qe_bias_label[i]),
                m_qe_bias_label->at(i)->size() * sizeof(crt_val_t));

            sgx_cudaMemcpyToDevice(
                m_dev_qe_bias_label[i],
                m_qe_bias_label->at(i)->get_components(),
                m_qe_bias_label->at(i)->size() * sizeof(crt_val_t));
        }
    }

    /**
     * @brief Evaluate garbled play layer on GPU.
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();

        int crt_base_size = m_gc->get_crt_base().size();

        size_t M = m_layer->get_output_dims().at(0);
        size_t K = m_layer->get_input_dims().at(0);
        auto dev_crt_base = m_gc->get_dev_crt_base();
        crt_val_t* crt_base = m_gc->get_crt_base().data();
        crt_val_t** dev_zero_label = m_gc->get_dev_zero_label();

        ocall_cuda_eval_dense(m_dev_out_label, dev_in_label, m_dev_qe_weights,
                              m_dev_qe_bias_label, dev_zero_label,
                              crt_base_size, M, K, crt_base);
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            auto crt_base = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(crt_base);
            size_t output_dim = m_layer->get_output_dims().at(0);
            dim_t dims{output_dim};
            m_out_label->at(i) = new LabelTensor(crt_base, dims);
            crt_val_t* comps = m_out_label->at(i)->get_components();
            size_t size = m_out_label->at(i)->size();
            crt_val_t* tmp;
            ocall_cudaMemcpyFromDevice(
                reinterpret_cast<void**>(&tmp), m_dev_out_label[i],
                output_dim * nr_comps * sizeof(crt_val_t));
            std::memcpy(comps, tmp, size * sizeof(crt_val_t));
            ocall_free(reinterpret_cast<void*>(tmp));
        }
    }

#endif

    ScalarTensor<q_val_t> get_qe_weights() { return m_qe_weights; }
    ScalarTensor<q_val_t> get_qe_bias() { return m_qe_biases; }
    crt_val_t** get_dev_out_label() override { return m_dev_out_label; }
    q_val_t* get_dev_qe_weights() { return m_dev_qe_weights; }
    crt_val_t** get_dev_qe_bias_label() { return m_dev_qe_bias_label; }

    vector<LabelTensor*>* get_qe_bias_label() { return m_qe_bias_label; }
    void set_qe_bias_label(vector<LabelTensor*>* qe_bias_label) {
        m_qe_bias_label = qe_bias_label;
    }
};

#endif
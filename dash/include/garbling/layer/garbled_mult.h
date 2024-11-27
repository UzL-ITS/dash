#ifndef GARBLED_MULT_H
#define GARBLED_MULT_H
// For testing purposes
// Multiply two inputs with the same modulus

#include <vector>
#include <cstdlib>

#include "garbling/gates/generalized_half_gate.h"
#include "garbling/layer/garbled_layer.h"
#include "circuit/layer/mult_layer.h"
#include "garbling/garbled_circuit_interface.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "crypto/cuda_aes_engine.h"
#include "misc/cuda_util.h"
#endif

using std::vector;

#ifndef SGX

__global__ void eval_mult(crt_val_t* dev_in_label, int output_size,
                          __uint128_t* dev_garbler_ciphers,
                          __uint128_t* dev_evaluator_ciphers,
                          crt_val_t* dev_out_label, crt_val_t modulus,
                          int nr_comps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        crt_val_t* out_label = &dev_out_label[idx * nr_comps];

        // Decrypt garbler half gate
        crt_val_t* sk01_xrR = &dev_in_label[(idx * 2) * nr_comps];
        auto garbler_label = new crt_val_t[nr_comps];

        eval_proj(garbler_label, sk01_xrR, &dev_garbler_ciphers[idx * modulus],
                  modulus, modulus, nr_comps, nr_comps);

        // Decrypt evaluator half gate
        crt_val_t* sk02_yR = &dev_in_label[(idx * 2 + 1) * nr_comps];
        eval_proj(out_label, sk02_yR, &dev_evaluator_ciphers[idx * modulus],
                  modulus, modulus, nr_comps, nr_comps);

        // Compute result
        // - Get color value
        crt_val_t y_plus_r = sk02_yR[0];
        // - compute output label
        for (int i = 0; i < nr_comps; ++i) {
            crt_val_t tmp = modulo(y_plus_r * sk01_xrR[i], modulus);
            out_label[i] = modulo(out_label[i] + tmp, modulus);
            out_label[i] = modulo(out_label[i] - garbler_label[i], modulus);
        }

        delete[] garbler_label;
    }
}

#endif

/**
 * @brief Computes product of input i and i+1 with i mod 2 = 0.
 *
 * Only supports crt bases of size 1.
 *
 */
class GarbledMult : public GarbledLayer {
    crt_val_t m_modulus;
    vector<GeneralizedHalfGate*> m_gen_hgs;
    __uint128_t* m_garbler_ciphers;
    __uint128_t* m_evaluator_ciphers;

#ifndef SGX
    __uint128_t* m_dev_garbler_ciphers{nullptr};
    __uint128_t* m_dev_evaluator_ciphers{nullptr};
    crt_val_t** m_dev_out_label{nullptr};
#endif

   public:
    GarbledMult(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc} {
        assert(gc->get_crt_base().size() == 1 &&
               "GarbledMult requires CRT base of size 1");
        assert(layer_ptr->get_input_size() % 2 == 0 &&
               "GarbledMult requires input size to be multiple of 2");
        assert(layer_ptr->get_output_size() ==
                   layer_ptr->get_input_size() / 2 &&
               "GarbledMult requires half as many outputs as inputs");

        m_modulus = gc->get_crt_base().at(0);
        int nr_ciphertexts = m_modulus * m_layer->get_output_size();

        m_garbler_ciphers = new __uint128_t[nr_ciphertexts];
        m_evaluator_ciphers = new __uint128_t[nr_ciphertexts];

        for (size_t i = 0; i < m_layer->get_output_size(); ++i) {
            auto gen_hg = new GeneralizedHalfGate(
                m_modulus, &m_garbler_ciphers[i * m_modulus],
                &m_evaluator_ciphers[i * m_modulus], gc);
            m_gen_hgs.push_back(gen_hg);
        }
    }

    virtual ~GarbledMult() {
        delete[] m_garbler_ciphers;
        delete[] m_evaluator_ciphers;
        for (auto gen_hg : m_gen_hgs) {
            delete gen_hg;
        }
#ifndef SGX
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                cudaCheckError(cudaFree(m_dev_out_label[i]));
            }
            delete[] m_dev_out_label;
        }
        if (m_dev_garbler_ciphers != nullptr) {
            cudaFree(m_dev_garbler_ciphers);
        }
        if (m_dev_evaluator_ciphers != nullptr) {
            cudaFree(m_dev_evaluator_ciphers);
        }
#endif
    }

    void garble(vector<LabelTensor*>* in_label) override {
        size_t output_size = m_layer->get_output_size();

        // Reserve output label tensor
        auto output_dims = m_layer->get_output_dims();
        m_out_label->resize(1);
        m_out_label->at(0) = new LabelTensor{m_modulus, output_dims};

        for (size_t i = 0; i < output_size; ++i) {
            auto label_a = in_label->at(0)->get_label(i * 2);
            auto label_b = in_label->at(0)->get_label(i * 2 + 1);
            auto result = m_gen_hgs.at(i)->garble(label_a, label_b);

            m_out_label->at(0)->set_label(result.get_label(0), i);
        }
    }

    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads) override {
        free_out_label();
        size_t output_size = m_layer->get_output_size();

        // Reserve output label tensor
        auto output_dims = m_layer->get_output_dims();
        m_out_label->resize(1);
        m_out_label->at(0) = new LabelTensor{m_modulus, output_dims};

        for (size_t i = 0; i < output_size; ++i) {
            auto label_a = encoded_inputs->at(0)->get_label(i * 2);
            auto label_b = encoded_inputs->at(0)->get_label(i * 2 + 1);
            auto result = m_gen_hgs.at(i)->cpu_evaluate(label_a, label_b);

            m_out_label->at(0)->set_label(result.get_label(0), i);
        }

        return m_out_label;
    }
#ifndef SGX
    void cuda_move() override {
        // Allocate memory for the output labels
        // - Allocate pointer array
        m_dev_out_label = new crt_val_t*[1];
        // - Allocate device array
        size_t output_size = m_layer->get_output_size();
        size_t nr_comps = LabelTensor::get_nr_comps(m_modulus);
        size_t size = output_size * nr_comps * sizeof(crt_val_t);
        cudaCheckError(cudaMalloc((void**)&m_dev_out_label[0], size));

        // Allocate memory for the garbler and evaluator half gates used in
        // multiplication
        cudaCheckError(
            cudaMalloc((void**)&m_dev_garbler_ciphers,
                       output_size * m_modulus * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_garbler_ciphers, m_garbler_ciphers,
                                  output_size * m_modulus * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));

        cudaCheckError(
            cudaMalloc((void**)&m_dev_evaluator_ciphers,
                       output_size * m_modulus * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_evaluator_ciphers, m_evaluator_ciphers,
                                  output_size * m_modulus * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));
    }

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();
        size_t nr_comps = LabelTensor::get_nr_comps(m_modulus);
        size_t output_size = m_layer->get_output_size();
        size_t nr_blocks = ceil_div(m_layer->get_output_size(), 32lu);

        eval_mult<<<nr_blocks, 32>>>(
            dev_in_label[0], output_size, m_dev_garbler_ciphers,
            m_dev_evaluator_ciphers, m_dev_out_label[0], m_modulus, nr_comps);
        cudaDeviceSynchronize();
    }

    crt_val_t** get_dev_out_label() override { return m_dev_out_label; }

    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());

        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            size_t nr_comps =
                LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
            int modulus = m_gc->get_crt_base().at(i);
            size_t output_size = m_layer->get_output_size();
            dim_t dims{m_layer->get_output_dims()};
            m_out_label->at(i) = new LabelTensor(modulus, dims);
            cudaCheckError(cudaMemcpy(
                m_out_label->at(i)->get_components(), m_dev_out_label[i],
                output_size * nr_comps * sizeof(crt_val_t),
                cudaMemcpyDeviceToHost));
        }
    }
#else
    /**
     * @brief Allocate memory for output labels on GPU.
     *
     */
    void cuda_move() override {
        throw std::runtime_error(
            "garbled_prod does not implement cuda_move() in SGX");
    }

    /**
     * @brief Evaluate garbled play layer on GPU.
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) override {
        throw std::runtime_error(
            "garbled_prod does not implement cuda_evaluate() in SGX");
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output() override {
        throw std::runtime_error(
            "garbled_prod does not implement cuda_move_output() in SGX");
    }

    /**
     * @brief Get pointer to the output labels of the layer on the GPU.
     *
     * @return T**
     */
    crt_val_t** get_dev_out_label() override {
        throw std::runtime_error(
            "garbled_prod does not implement get_dev_out_label() in SGX");
    }
#endif
};

#endif
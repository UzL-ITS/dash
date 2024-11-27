#ifndef GARBLED_MIXED_MOD_MULT_H
#define GARBLED_MIXED_MOD_MULT_H
// For testing purposes
// Multiply two inputs with different moduli

#include <cstdlib>
#include <vector>

#include "circuit/layer/mixed_mod_mult_layer.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/mixed_mod_half_gate.h"
#include "garbling/gates/projection_gate.h"
#include "garbling/layer/garbled_layer.h"
#include "misc/misc.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "crypto/cuda_aes_engine.h"
#include "misc/cuda_util.h"
#endif

using std::vector;

#ifndef SGX

__global__ void eval_mixed_mod_mult_dummy(
    crt_val_t* dev_in_label, int output_size,
    __uint128_t* dev_mod_transformation_ciphers,
    __uint128_t* dev_garbler_ciphers, __uint128_t* dev_evaluator_ciphers,
    crt_val_t* dev_out_label, crt_val_t p, crt_val_t q, int nr_comps_p,
    int nr_comps_q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        // Transfer second input to smaller modulus
        auto mod_transformed_input = new crt_val_t[nr_comps_q];
        auto input2 = &dev_in_label[(idx * 2 + 1) * nr_comps_p];
        eval_proj(mod_transformed_input, input2,
                  &dev_mod_transformation_ciphers[idx * p], p, q, nr_comps_p,
                  nr_comps_q);

        auto out_label = &dev_out_label[idx * nr_comps_p];
        auto dev_garbler_label = &dev_in_label[(idx * 2) * nr_comps_p];
        __uint128_t* g_ciphers = &dev_garbler_ciphers[idx * p];
        __uint128_t* e_ciphers = &dev_evaluator_ciphers[idx * (q + 1)];

        eval_mixed_mod_mult(out_label, dev_garbler_label, mod_transformed_input,
                            g_ciphers, e_ciphers, p, q, nr_comps_p, nr_comps_q);

        free(mod_transformed_input);
    }
}

#endif

class GarbledMixedModMult : public GarbledLayer {
    MixedModMultLayer* m_mult;
    vector<MixedModHalfGate*> m_mixed_mod_hgs;
    __uint128_t* m_garbler_ciphers;
    __uint128_t* m_evaluator_ciphers;

    crt_val_t m_modulus_p;
    crt_val_t m_modulus_q;  // Smaller modulus

    // Transfer second input to smaller_modulus
    vector<ProjectionGate*> m_mod_transformations;
    __uint128_t* m_mod_transformation_ciphers;

#ifndef SGX
    __uint128_t* m_dev_garbler_ciphers{nullptr};
    __uint128_t* m_dev_evaluator_ciphers{nullptr};
    __uint128_t* m_dev_mod_transformation_ciphers{nullptr};
    crt_val_t** m_dev_out_label{nullptr};
#endif

   public:
    GarbledMixedModMult(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc} {
        assert(gc->get_crt_base().size() == 1 &&
               "GarbledMixedModMult requires CRT base of size 1");
        assert(layer_ptr->get_input_size() % 2 == 0 &&
               "GarbledMixedModMult requires input size to be multiple of 2");
        assert(layer_ptr->get_output_size() ==
                   layer_ptr->get_input_size() / 2 &&
               "GarbledMixedModMult requires half as many outputs as inputs");

        m_mult = static_cast<MixedModMultLayer*>(layer_ptr);

        m_modulus_q = m_mult->get_smaller_modulus();
        m_modulus_p = m_gc->get_crt_base().at(0);

        size_t nr_outputs = m_mult->get_output_size();

        // Create Projection Gate to reduce second input to smaller modulus
        int size = nr_outputs * m_modulus_p;
        m_mod_transformation_ciphers = new __uint128_t[size];
        for (size_t i = 0; i < nr_outputs; ++i) {
            auto cipher = &m_mod_transformation_ciphers[i * m_modulus_p];
            auto func = &ProjectionFunctionalities::identity;
            auto p = new ProjectionGate(m_modulus_q, cipher, func, gc);
            m_mod_transformations.push_back(p);
        }

        // Create MixedModHalfGates
        m_garbler_ciphers = new __uint128_t[nr_outputs * m_modulus_p];
        m_evaluator_ciphers = new __uint128_t[nr_outputs * (m_modulus_q + 1)];

        for (size_t i = 0; i < nr_outputs; ++i) {
            auto g_c = &m_garbler_ciphers[i * m_modulus_p];
            auto e_c = &m_evaluator_ciphers[i * (m_modulus_q + 1)];
            auto mmhg =
                new MixedModHalfGate(m_modulus_q, m_modulus_p, g_c, e_c, gc);
            m_mixed_mod_hgs.push_back(mmhg);
        }
    }

    virtual ~GarbledMixedModMult() {
        for (auto& p : m_mod_transformations) {
            delete p;
        }
        for (auto& mmhg : m_mixed_mod_hgs) {
            delete mmhg;
        }
        delete[] m_mod_transformation_ciphers;
        delete[] m_garbler_ciphers;
        delete[] m_evaluator_ciphers;
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
        if (m_dev_mod_transformation_ciphers != nullptr) {
            cudaFree(m_dev_mod_transformation_ciphers);
        }
#endif
    }

    void garble(vector<LabelTensor*>* in_label) override {
        // Reserve output label tensor
        auto output_dims = m_layer->get_output_dims();
        m_out_label->resize(1);
        m_out_label->at(0) = new LabelTensor{m_modulus_p, output_dims};

        size_t output_size = m_mult->get_output_size();

        for (size_t i = 0; i < output_size; ++i) {
            // Transfer second input to smaller_modulus
            LabelTensor out_base_label{m_modulus_q};
            out_base_label.init_random();
            auto out_offset_label = m_gc->get_label_offset(m_modulus_q);
            m_mod_transformations.at(i)->garble(
                in_label->at(0)->get_label(i * 2 + 1), &out_base_label,
                out_offset_label);

            // Garble mixed modulus half gate for the multiplication
            auto result = m_mixed_mod_hgs.at(i)->garble(
                in_label->at(0)->get_label(i * 2), out_base_label.get_label(0));

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
        m_out_label->at(0) = new LabelTensor{m_modulus_p, output_dims};

        for (size_t i = 0; i < output_size; ++i) {
            // Transfer second input to smaller_modulus
            auto mod_tranformed_input =
                m_mod_transformations.at(i)->cpu_evaluate(
                    encoded_inputs->at(0)->get_label(i * 2 + 1));

            auto result = m_mixed_mod_hgs.at(i)->cpu_evaluate(
                encoded_inputs->at(0)->get_label(i * 2),
                mod_tranformed_input.get_label(0));

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
        size_t nr_comps = LabelTensor::get_nr_comps(m_modulus_p);
        size_t size = output_size * nr_comps * sizeof(crt_val_t);
        cudaCheckError(cudaMalloc((void**)&m_dev_out_label[0], size));

        // Allocate memory for the garbler and evaluator half gates used in
        // multiplication
        size = output_size * m_modulus_p * sizeof(__uint128_t);
        cudaCheckError(cudaMalloc((void**)&m_dev_garbler_ciphers, size));
        cudaCheckError(cudaMemcpy(m_dev_garbler_ciphers, m_garbler_ciphers,
                                  size, cudaMemcpyHostToDevice));

        size = output_size * (m_modulus_q + 1) * sizeof(__uint128_t);
        cudaCheckError(cudaMalloc((void**)&m_dev_evaluator_ciphers, size));
        cudaCheckError(cudaMemcpy(m_dev_evaluator_ciphers, m_evaluator_ciphers,
                                  size, cudaMemcpyHostToDevice));

        // Allocate memory for the transformation gate used to transfer the
        // second input to the smaller modulus
        size = output_size * m_modulus_p * sizeof(__uint128_t);
        cudaCheckError(
            cudaMalloc((void**)&m_dev_mod_transformation_ciphers, size));
        cudaCheckError(cudaMemcpy(m_dev_mod_transformation_ciphers,
                                  m_mod_transformation_ciphers, size,
                                  cudaMemcpyHostToDevice));
    }

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();
        size_t nr_comps_p = LabelTensor::get_nr_comps(m_modulus_p);
        size_t nr_comps_q = LabelTensor::get_nr_comps(m_modulus_q);
        size_t output_size = m_layer->get_output_size();
        size_t nr_blocks = ceil_div(m_layer->get_output_size(), 32lu);

        eval_mixed_mod_mult_dummy<<<nr_blocks, 32>>>(
            dev_in_label[0], output_size, m_dev_mod_transformation_ciphers,
            m_dev_garbler_ciphers, m_dev_evaluator_ciphers, m_dev_out_label[0],
            m_modulus_p, m_modulus_q, nr_comps_p, nr_comps_q);
        cudaDeviceSynchronize();
    }

    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());
        size_t output_size = m_layer->get_output_size();

        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            int modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            dim_t dims{m_layer->get_output_dims()};
            m_out_label->at(i) = new LabelTensor(modulus, dims);

            cudaCheckError(cudaMemcpy(
                m_out_label->at(i)->get_components(), m_dev_out_label[i],
                output_size * nr_comps * sizeof(crt_val_t),
                cudaMemcpyDeviceToHost));
        }
    }

    crt_val_t** get_dev_out_label() override { return m_dev_out_label; }
#else
    /**
     * @brief Allocate memory for output labels on GPU.
     *
     */
    void cuda_move() override {
        throw std::runtime_error(
            "garbled_mixed_mod_mult does not implement cuda_move() in SGX");
    }

    /**
     * @brief Evaluate garbled play layer on GPU.
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) override {
        throw std::runtime_error(
            "garbled_mixed_mod_mult does not implement cuda_evaluate() in SGX");
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output() override {
        throw std::runtime_error(
            "garbled_mixed_mod_mult does not implement cuda_move_output() in "
            "SGX");
    }

    /**
     * @brief Get pointer to the output labels of the layer on the GPU.
     *
     * @return T**
     */
    crt_val_t** get_dev_out_label() override {
        throw std::runtime_error(
            "garbled_mixed_mod_mult does not implement get_dev_out_label() in "
            "SGX");
    }
#endif
};

#endif
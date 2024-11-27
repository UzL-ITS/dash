#ifndef GARBLED_RELU_H
#define GARBLED_RELU_H

#include <cstdlib>
#include <vector>

#include "circuit/layer/relu.h"
#include "garbling/gadgets/sign_gadget.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/mixed_mod_half_gate.h"
#include "garbling/gates/projection_gate.h"
#include "garbling/layer/garbled_layer.h"
#include "misc/datatypes.h"
#include "misc/util.h"

using std::vector;

#ifndef SGX
__global__ void eval_sign_mult(
    crt_val_t* out_labels, crt_val_t* dev_garbler_labels,
    crt_val_t* dev_eval_labels, __uint128_t* dev_garbler_ciphers,
    __uint128_t* dev_eval_ciphers, crt_val_t p, crt_val_t q, int nr_comps_p,
    int nr_comps_q, size_t nr_inputs, size_t crt_base_size, int crt_base_sum,
    int crt_base_prefix, size_t crt_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nr_inputs) {
        auto o_labels = out_labels + idx * nr_comps_p;
        auto g_label = dev_garbler_labels + idx * nr_comps_p;
        auto e_label = dev_eval_labels + idx * nr_comps_q;
        auto g_c = dev_garbler_ciphers + idx * crt_base_sum + crt_base_prefix;
        auto e_c = dev_eval_ciphers + idx * crt_base_size * (q + 1) +
                   crt_idx * (q + 1);

        eval_mixed_mod_mult(o_labels, g_label, e_label, g_c, e_c, p, q,
                            nr_comps_p, nr_comps_q);
    }
}
#endif

class GarbledRelu : public GarbledLayer {
    SignGadget* m_sign_gadget;

    vector<int> cipher_sizes;

    // Step 4: Mixed-mod-halfgates to multiply the inputs with the sign bits
    //// size: nr_inputs x crt_base_size
    vector<vector<MixedModHalfGate*>> m_mixed_mod_half_gates;
    /// size: nr_inputs * crt_base_sum
    __uint128_t* m_garbler_ciphers;
    //// size: crt_base_size * nr_inputs * (q + 1), with q = 2
    __uint128_t* m_evaluator_ciphers;

    crt_val_t** m_dev_out_label{nullptr};
    __uint128_t* m_dev_garbler_ciphers{nullptr};
    __uint128_t* m_dev_evaluator_ciphers{nullptr};

   public:
    GarbledRelu(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc} {
        size_t nr_inputs = m_layer->get_input_size();
        size_t crt_base_size = m_gc->get_crt_base().size();

        vector<crt_val_t> out_moduli{2};
        m_sign_gadget = new SignGadget{m_gc, 0, 1, nr_inputs, out_moduli};

        // Reserve memory for mixed-mod-halfgates
        int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                           m_gc->get_crt_base().end(), 0);
        crt_val_t q = 2;
        size_t size = nr_inputs * crt_base_sum;
        cipher_sizes.push_back(size);
        m_garbler_ciphers = new __uint128_t[size];
        size = nr_inputs * crt_base_size * (q + 1);
        cipher_sizes.push_back(size);
        m_evaluator_ciphers = new __uint128_t[size];
    }

    virtual ~GarbledRelu() {
        delete m_sign_gadget;
        for (auto& vec : m_mixed_mod_half_gates) {
            for (auto& gate : vec) {
                delete gate;
            }
        }
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
            cudaCheckError(cudaFree(m_dev_garbler_ciphers));
        }
        if (m_dev_evaluator_ciphers != nullptr) {
            cudaCheckError(cudaFree(m_dev_evaluator_ciphers));
        }
#else
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                ocall_cudaFree(m_dev_out_label[i]);
            }
            ocall_free(m_dev_out_label);
        }

        if (m_dev_garbler_ciphers != nullptr) {
            ocall_cudaFree(m_dev_garbler_ciphers);
        }
        if (m_dev_evaluator_ciphers != nullptr) {
            ocall_cudaFree(m_dev_evaluator_ciphers);
        }
#endif
    }

    void garble(vector<LabelTensor*>* in_label) override {
        size_t crt_base_size = m_gc->get_crt_base().size();  // k
        dim_t output_dims = m_layer->get_output_dims();
        size_t input_size = m_layer->get_input_size();

        int mrs_base_sum_suffix = std::accumulate(
            m_gc->get_mrs_base().begin() + 1, m_gc->get_mrs_base().end(), 0);

        // Reserve ouput_label
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        // Step 1 - 3
        //// Reserve out label for sign gagdet
        vector<LabelTensor*> out_label_sign;
        auto label = new LabelTensor{2, output_dims};
        label->init_random();
        out_label_sign.push_back(label);

        m_sign_gadget->garble(in_label, &out_label_sign);

        // Step 4: Multiply sign bit with input
        m_mixed_mod_half_gates.resize(input_size);
#ifndef SGX
#pragma omp parallel for
#endif
        for (size_t i = 0; i < input_size; ++i) {
            auto crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                                m_gc->get_crt_base().end(), 0);
            int crt_base_prefix = 0;
            vector<MixedModHalfGate*> mm_hgs;
            for (size_t j = 0; j < crt_base_size; ++j) {
                crt_val_t q = 2;
                crt_val_t p = m_gc->get_crt_base().at(j);

                //// Get cipher indices
                int idx = i * crt_base_sum + crt_base_prefix;
                __uint128_t* g_c = &m_garbler_ciphers[idx];
                idx = i * crt_base_size * (q + 1) + j * (q + 1);
                __uint128_t* e_c = &m_evaluator_ciphers[idx];

                auto mmhg = new MixedModHalfGate{q, p, g_c, e_c, m_gc};
                mm_hgs.push_back(mmhg);

                auto tmp = mmhg->garble(in_label->at(j)->get_label(i),
                                        out_label_sign.at(0)->get_label(i));
                m_out_label->at(j)->set_label(tmp.get_label(0), i);

                crt_base_prefix += p;
            }
            m_mixed_mod_half_gates.at(i) = mm_hgs;
        }

        // clean up
        for (auto label : out_label_sign) {
            delete label;
        }
    }

    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads) override {
        size_t crt_base_size = m_gc->get_crt_base().size();  // k
        dim_t output_dims = m_layer->get_output_dims();

        // Reserve ouput_label
        free_out_label();
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        // Step 1 - 3
        //// Reserve out label for sign gagdet
        vector<LabelTensor*> out_label_sign;
        auto label = new LabelTensor{2, output_dims};
        out_label_sign.push_back(label);

        m_sign_gadget->cpu_evaluate(encoded_inputs, &out_label_sign,
                                    nr_threads);

#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
        for (size_t i = 0; i < m_layer->get_input_size(); ++i) {
            // Step 4: Multiply sign bit with input
            for (size_t j = 0; j < crt_base_size; ++j) {
                auto mm_hg = m_mixed_mod_half_gates.at(i).at(j);
                auto result =
                    mm_hg->cpu_evaluate(encoded_inputs->at(j)->get_label(i),
                                        out_label_sign.at(0)->get_label(i));
                m_out_label->at(j)->set_label(result.get_label(0), i);
            }
        }

        // clean up
        for (auto label : out_label_sign) {
            delete label;
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
            size_t nr_comps =
                LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
            size_t size = output_size * nr_comps * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_out_label[i]), size));
        }

        // Step 1 - 3
        m_sign_gadget->cuda_move();

        // Move ciphertexts for step 4 to device
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_garbler_ciphers),
                       cipher_sizes.at(0) * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_garbler_ciphers, m_garbler_ciphers,
                                  cipher_sizes.at(0) * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_evaluator_ciphers),
                       cipher_sizes.at(1) * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_evaluator_ciphers, m_evaluator_ciphers,
                                  cipher_sizes.at(1) * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));
    }

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();
        size_t crt_base_size = m_gc->get_crt_base().size();
        cudaStream_t stream[crt_base_size];

        size_t nr_blocks = ceil_div(m_layer->get_output_size(), 32lu);

        // Step 1 - 3
        m_sign_gadget->cuda_evaluate(dev_in_label);
        auto dev_out_label_sign = m_sign_gadget->get_dev_out_label();

        // Step 4: Multiply sign bit with input
        size_t input_size = m_layer->get_input_size();

        int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                           m_gc->get_crt_base().end(), 0);
        int crt_base_prefix = 0;
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t q = 2;
            crt_val_t p = m_gc->get_crt_base().at(i);
            size_t nr_comps_q = LabelTensor::get_nr_comps(q);
            size_t nr_comps_p = LabelTensor::get_nr_comps(p);
            size_t output_size = m_layer->get_output_size();

            cudaCheckError(cudaStreamCreate(&stream[i]));

            eval_sign_mult<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_out_label[i], dev_in_label[i], dev_out_label_sign[0],
                m_dev_garbler_ciphers, m_dev_evaluator_ciphers, p, q,
                nr_comps_p, nr_comps_q, input_size, crt_base_size, crt_base_sum,
                crt_base_prefix, i);

            cudaCheckError(cudaStreamDestroy(stream[i]));
            crt_base_prefix += m_gc->get_crt_base().at(i);
        }
        cudaDeviceSynchronize();
    }

    void cuda_move_output() override {
        auto crt_base_size = m_gc->get_crt_base().size();
        m_out_label->resize(crt_base_size);

        for (size_t i = 0; i < crt_base_size; ++i) {
            auto crt_base = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(crt_base);
            auto output_dims = m_layer->get_output_dims();
            auto output_size = m_layer->get_output_size();
            m_out_label->at(i) = new LabelTensor(crt_base, output_dims);
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
        // Allocate memory for the output labels
        // - Allocate pointer array
        int size = m_gc->get_crt_base().size() * sizeof(crt_val_t*);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_out_label),
                              size);

        // - Allocate device array
        size_t output_size = m_layer->get_output_size();
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            int nr_comps = LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
            size_t size = output_size * nr_comps * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_out_label[i]),
                             size);
        }

        // Step 1 - 3
        m_sign_gadget->cuda_move();

        // Move ciphertexts for step 4 to device
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_garbler_ciphers),
                         cipher_sizes.at(0) * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_garbler_ciphers, m_garbler_ciphers,
                               cipher_sizes.at(0) * sizeof(__uint128_t));

        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_evaluator_ciphers),
                         cipher_sizes.at(1) * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_evaluator_ciphers, m_evaluator_ciphers,
                               cipher_sizes.at(1) * sizeof(__uint128_t));
    }

    /**
     * @brief Evaluate garbled play layer on GPU.
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();

        // Step 1 - 3
        m_sign_gadget->cuda_evaluate(dev_in_label);
        auto dev_out_label_sign = m_sign_gadget->get_dev_out_label();

        // Step 4
        size_t input_size = m_layer->get_input_size();
        size_t crt_base_size = m_gc->get_crt_base().size();
        int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                           m_gc->get_crt_base().end(), 0);
        ocall_sign_mult(m_dev_out_label, dev_in_label, dev_out_label_sign,
                        m_dev_garbler_ciphers, m_dev_evaluator_ciphers,
                        input_size, crt_base_size, crt_base_sum,
                        m_gc->get_crt_base().data());
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
            memcpy(comps, tmp, size * sizeof(crt_val_t));
            ocall_free(reinterpret_cast<void*>(tmp));
        }
    }
#endif
    crt_val_t** get_dev_out_label() override { return m_dev_out_label; }
};

#endif
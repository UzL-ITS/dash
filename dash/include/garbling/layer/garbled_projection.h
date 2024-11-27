#ifndef GARBLED_PROJECTION_H
#define GARBLED_PROJECTION_H
// For testing purposes

#include <vector>
#include <cstdlib>

#include "garbling/layer/garbled_layer.h"
#include "circuit/layer/projection.h"
#include "garbling/gates/projection_gate.h"
#include "garbling/garbled_circuit_interface.h"
#include "misc/util.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "misc/cuda_util.h"
#endif

using std::vector;

class GarbledProjection : public GarbledLayer {
    Projection* m_projection;
    vector<vector<ProjectionGate*>> m_gates;
    vector<__uint128_t*> m_ciphers;

#ifndef SGX
    crt_val_t** m_dev_out_label{nullptr};
    __uint128_t** m_dev_ciphers{nullptr};
#endif

   public:
    GarbledProjection(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc},
          m_projection{static_cast<Projection*>(layer_ptr)} {
        size_t input_size = m_projection->get_input_size();

        // get_in_moduli.size() == get_crt_base.size()
        for (size_t i = 0; i < m_projection->get_in_moduli().size(); ++i) {
            crt_val_t in_modulus = m_projection->get_in_moduli().at(i);
            crt_val_t out_modulus = m_projection->get_out_moduli().at(i);

            int nr_ciphertexts = input_size * in_modulus;
            __uint128_t* ciphers = new __uint128_t[nr_ciphertexts];
            m_ciphers.push_back(ciphers);

            vector<ProjectionGate*> gates;
            gates.reserve(input_size);
            for (size_t j = 0; j < input_size; ++j) {
                auto cipher = &ciphers[j * in_modulus];
                gates.push_back(new ProjectionGate{
                    out_modulus, cipher, m_projection->get_functionality(),
                    m_gc});
            }
            m_gates.push_back(gates);
        }
    }

    virtual ~GarbledProjection() {
        for (auto& cipher : m_ciphers) {
            delete[] cipher;
        }
        for (auto& gates : m_gates) {
            for (auto& gate : gates) {
                delete gate;
            }
        }
#ifndef SGX
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                cudaCheckError(cudaFree(m_dev_out_label[i]));
            }
            delete[] m_dev_out_label;
        }

        if (m_dev_ciphers != nullptr) {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
                cudaCheckError(cudaFree(m_dev_ciphers[i]));
            }
            delete[] m_dev_ciphers;
        }
#endif
    }

#ifdef GRR3
    void garble(vector<LabelTensor*>* in_label) override {
        m_out_label->resize(m_gc->get_crt_base().size());

        size_t input_size = m_projection->get_input_size();
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            m_out_label->at(i) = new LabelTensor(*in_label->at(i));
            for (int j = 0; j < m_projection->get_output_size(); ++j) {
                __uint128_t tmp =
                    m_gates.at(i).at(j)->garble(in_label->at(i), j);
                m_out_label->at(i)->set_compressed(tmp, j);
            }
            m_out_label->at(i)->decompress();
        }
    }
#else
    void garble(vector<LabelTensor*>* in_label) override {
        m_out_label->resize(m_gc->get_crt_base().size());

        size_t input_size = m_projection->get_input_size();
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            dim_t dims = m_projection->get_output_dims();
            crt_val_t out_modulus = m_projection->get_out_moduli().at(i);
            m_out_label->at(i) = new LabelTensor{out_modulus, dims};
            for (size_t j = 0; j < m_projection->get_output_size(); ++j) {
                auto l = in_label->at(i)->get_label(j);
                auto tmp = m_gates.at(i).at(j)->garble(l);
                m_out_label->at(i)->set_compressed(tmp, j);
            }
            m_out_label->at(i)->decompress();
        }
    }
#endif
    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads) override {
        free_out_label();
        m_out_label->resize(m_gc->get_crt_base().size());
        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            m_out_label->at(i) =
                new LabelTensor(m_projection->get_out_moduli().at(i),
                                m_layer->get_output_dims());
            for (size_t j = 0; j < m_projection->get_output_size(); ++j) {
                auto l = encoded_inputs->at(i)->get_label(j);
                auto out_label = m_gates.at(i).at(j)->cpu_evaluate(l);
                m_out_label->at(i)->set_label(out_label.get_label(0), j);
            }
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
                LabelTensor::get_nr_comps(m_projection->get_out_moduli().at(i));
            size_t size = output_size * nr_comps * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc((void**)&m_dev_out_label[i], size));
        }

        // Move ciphertexts
        // - Allocate pointer array
        size_t input_size = m_projection->get_input_size();
        // -- get_in_moduli.size() == get_crt_base.size()
        m_dev_ciphers = new __uint128_t*[m_projection->get_in_moduli().size()];
        // - Allocate device array
        for (size_t i = 0; i < m_projection->get_in_moduli().size(); ++i) {
            crt_val_t in_modulus = m_projection->get_in_moduli().at(i);

            int nr_ciphertexts = input_size * in_modulus;
            cudaCheckError(cudaMalloc((void**)&m_dev_ciphers[i],
                                      nr_ciphertexts * sizeof(__uint128_t)));
            cudaCheckError(cudaMemcpy(m_dev_ciphers[i], m_ciphers[i],
                                      nr_ciphertexts * sizeof(__uint128_t),
                                      cudaMemcpyHostToDevice));
        }
    };

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();

        int nr_streams = m_gc->get_crt_base().size();
        cudaStream_t stream[nr_streams];

        size_t nr_blocks = ceil_div(m_layer->get_output_size(), 32lu);

        size_t nr_inputs = m_projection->get_input_size();

        for (int i = 0; i < nr_streams; ++i) {
            cudaCheckError(cudaStreamCreate(&stream[i]));

            crt_val_t in_modulus = m_projection->get_in_moduli().at(i);
            size_t nr_comps_in = LabelTensor::get_nr_comps(in_modulus);
            crt_val_t out_modulus = m_projection->get_out_moduli().at(i);
            size_t nr_comps_out = LabelTensor::get_nr_comps(out_modulus);

            eval_proj<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_out_label[i], dev_in_label[i], m_dev_ciphers[i],
                m_projection->get_in_moduli().at(i),
                m_projection->get_out_moduli().at(i), nr_comps_in, nr_comps_out,
                nr_inputs);
            cudaCheckError(cudaStreamDestroy(stream[i]));
        }
        cudaDeviceSynchronize();
    };

    void cuda_move_output() override {
        m_out_label->resize(m_gc->get_crt_base().size());

        for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i) {
            size_t nr_comps =
                LabelTensor::get_nr_comps(m_gc->get_crt_base().at(i));
            int modulus = m_gc->get_crt_base().at(i);
            dim_t dims = m_layer->get_output_dims();
            size_t output_size = m_layer->get_output_size();
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
            "garbled_projection does not implement cuda_move() in SGX");
    }

    /**
     * @brief Evaluate garbled play layer on GPU.
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) override {
        throw std::runtime_error(
            "garbled_projection does not implement cuda_evaluate() in SGX");
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output() override {
        throw std::runtime_error(
            "garbled_projection does not implement cuda_move_output() in SGX");
    }

    /**
     * @brief Get pointer to the output labels of the layer on the GPU.
     *
     * @return T**
     */
    crt_val_t** get_dev_out_label() override {
        throw std::runtime_error(
            "garbled_projection does not implement get_dev_out_label() in SGX");
    }

#endif
};

#endif
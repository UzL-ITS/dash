#ifndef GARBERED_CIRCUIT_H
#define GARBERED_CIRCUIT_H

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "circuit/circuit.h"
#include "circuit/scalar_tensor.h"
#include "garbling/gadgets/lookup_approx_sign.h"
#include "garbling/label_tensor.h"
#include "garbling/layer/garbled_layer.h"
#include "misc/datatypes.h"
#include "misc/util.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "misc/cuda_util.h"
#endif

using std::vector;

#ifdef BENCHMARK
#include <chrono>
#include <map>

#include "circuit/layer/layer.h"
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
#endif

class GarbledCircuitInterface {
#ifdef BENCHMARK
    std::map<std::string, double> m_eval_times;
    std::map<Layer::LayerType, std::string> m_layer_names{
        {Layer::dense, "dense"},
        {Layer::conv2d, "conv2d"},
        {Layer::sum_layer, "sum_layer"},
        {Layer::projection, "projection"},
        {Layer::mult_layer, "mult_layer"},
        {Layer::mixed_mod_mult_layer, "mixed_mod_mult_layer"},
        {Layer::approx_relu, "approx_relu"},
        {Layer::flatten, "flatten"},
        {Layer::sign, "sign"},
        {Layer::max_pool, "max_pool"},
        {Layer::max, "max"},
        {Layer::rescale, "rescale"},
        {Layer::base_extension, "base_extension"}
    };
#endif
   protected:
    Circuit* m_circuit{nullptr};
    // moduli for chinese remainer representation
    vector<crt_val_t> m_crt_base{};
    // maximal modulus usable in circuit
    crt_val_t m_max_modulus;
    // product of all the crt_base values
    q_val_t m_crt_modulus;
    // moduli for mixed radix representation used in the approx. sign gate
    vector<mrs_val_t> m_mrs_base{};
    vector<GarbledLayer*> m_garbled_layer{};
    // crt_moduli x range(crt_moduli_i) x output_dim
    vector<vector<ScalarTensor<__uint128_t>>> m_decoding_information{};

    // One Labeltensor per crt modulus (each LabelTensor number of
    // circuit-inputs labels)
    vector<LabelTensor*> m_base_label{};

    // One label for each modulus from 2 ... max_modulus
    // - label representing zero value
    vector<LabelTensor*> m_zero_label{};
    // - label offsets
    vector<LabelTensor*> m_label_offset{};

    // Lookup table for approx. sign gate
    LookupApproxSign* m_lookup_approx_sign{nullptr};

    // Constant input labels for garbled rescaling
    //// Upschift labels
    vector<LabelTensor*> m_upshift_labels_base{};
    vector<LabelTensor*> m_upshift_labels{};
    //// Downshift labels
    std::map<crt_val_t, std::vector<LabelTensor*>> m_downshift_labels_base{};
    std::map<crt_val_t, std::vector<LabelTensor*>> m_downshift_labels{};

    mrs_val_t* m_dev_mrs_base{nullptr};
    crt_val_t* m_dev_crt_base{nullptr};
    crt_val_t** m_dev_zero_label{nullptr};
    // pointer array on device
    crt_val_t** m_dev_dev_zero_label{nullptr};

    crt_val_t** m_dev_upshift_labels{nullptr};
    crt_val_t** m_dev_dev_upshift_labels{nullptr};
    crt_val_t** m_dev_downshift_labels{nullptr};
    crt_val_t** m_dev_dev_downshift_labels{nullptr};

   public:
    GarbledCircuitInterface(Circuit* circuit, int cpm_size,
                            crt_val_t max_modulus = 0, bool garble_me = true)
        : m_circuit{circuit},
          m_crt_base{util::sieve_of_eratosthenes<crt_val_t>(cpm_size)},
          m_max_modulus{m_crt_base.back()},
          m_crt_modulus{std::accumulate(begin(m_crt_base), end(m_crt_base), 1ll,
                                        std::multiplies<q_val_t>())} {
        if (max_modulus > m_max_modulus) {
            m_max_modulus = max_modulus;
        } else if (max_modulus != 0) {
#ifndef SGX
            std::cerr << "Given max. modulus is smaller than largest modulus "
                         "of CRT-Base. Set max. modulus = max(CRT-BASE)!"
                      << std::endl;
#endif
        }
    }

    GarbledCircuitInterface(Circuit* circuit, int cpm_size,
                            const vector<mrs_val_t>& mrs_base,
                            crt_val_t max_modulus = 0, bool garble_me = true)
        : m_circuit{circuit},
          m_crt_base{util::sieve_of_eratosthenes<crt_val_t>(cpm_size)},
          m_max_modulus{compute_max_modulus(m_crt_base, mrs_base)},
          m_crt_modulus{std::accumulate(begin(m_crt_base), end(m_crt_base), 1ll,
                                        std::multiplies<q_val_t>())},
          m_mrs_base{mrs_base},
          m_lookup_approx_sign{new LookupApproxSign(m_crt_base, m_mrs_base)} {
        if (max_modulus > m_max_modulus) {
            m_max_modulus = max_modulus;
        } else if (max_modulus != 0) {
#ifndef SGX
            std::cerr << "Given max. modulus is smaller than largest modulus "
                         "of CRT-Base. Set max. modulus = max(CRT-BASE)!"
                      << std::endl;
#endif
        }
    }

    GarbledCircuitInterface(Circuit* circuit, int cpm_size, float mrs_accuracy,
                            crt_val_t max_modulus = 0, bool garble_me = true)
        : m_circuit{circuit},
          m_crt_base{util::sieve_of_eratosthenes<crt_val_t>(cpm_size)},
          m_max_modulus{compute_max_modulus(
              m_crt_base, get_mrs_base(cpm_size, mrs_accuracy))},
          m_crt_modulus{std::accumulate(begin(m_crt_base), end(m_crt_base), 1ll,
                                        std::multiplies<q_val_t>())},
          m_mrs_base{get_mrs_base(cpm_size, mrs_accuracy)},
          m_lookup_approx_sign{new LookupApproxSign(m_crt_base, m_mrs_base)} {
        if (max_modulus > m_max_modulus) {
            m_max_modulus = max_modulus;
        } else if (max_modulus != 0) {
#ifndef SGX
            std::cerr << "Given max. modulus is smaller than largest modulus "
                         "of CRT-Base. Set max. modulus = max(CRT-BASE)!"
                      << std::endl;
#endif
        }
    }

    GarbledCircuitInterface(Circuit* circuit, const vector<crt_val_t>& crt_base,
                            crt_val_t max_modulus = 0, bool garble_me = true)
        : m_circuit{circuit},
          m_crt_base{crt_base},
          m_max_modulus{m_crt_base.back()},
          m_crt_modulus{std::accumulate(begin(m_crt_base), end(m_crt_base), 1ll,
                                        std::multiplies<q_val_t>())} {
        if (max_modulus > m_max_modulus) {
            m_max_modulus = max_modulus;
        } else if (max_modulus != 0) {
#ifndef SGX
            std::cerr << "Given max. modulus is smaller than largest modulus "
                         "of CRT-Base. Set max. modulus = max(CRT-BASE)!"
                      << std::endl;
#endif
        }
    }

    GarbledCircuitInterface(Circuit* circuit, const vector<crt_val_t>& crt_base,
                            const vector<mrs_val_t>& mrs_base,
                            crt_val_t max_modulus = 0, bool garble_me = true)
        : m_circuit{circuit},
          m_crt_base{crt_base},
          m_max_modulus{compute_max_modulus(m_crt_base, mrs_base)},
          m_crt_modulus{std::accumulate(begin(m_crt_base), end(m_crt_base), 1ll,
                                        std::multiplies<q_val_t>())},
          m_mrs_base{mrs_base},
          m_lookup_approx_sign{new LookupApproxSign(m_crt_base, m_mrs_base)} {
        if (max_modulus > m_max_modulus) {
            m_max_modulus = max_modulus;
        } else if (max_modulus != 0) {
#ifndef SGX
            std::cerr << "Given max. modulus is smaller than largest modulus "
                         "of CRT-Base. Set max. modulus = max(CRT-BASE)!"
                      << std::endl;
#endif
        }
    }

    virtual ~GarbledCircuitInterface() {
        for (auto layer : m_garbled_layer) {
            delete layer;
        }
        for (auto label : m_label_offset) {
            delete label;
        }
        for (auto label : m_zero_label) {
            delete label;
        }
        for (auto& label : m_base_label) {
            delete label;
        }
        delete m_lookup_approx_sign;
        for (auto label : m_upshift_labels_base) {
            delete label;
        }
        for (auto label : m_upshift_labels) {
            delete label;
        }
        for (auto pair : m_downshift_labels_base) {
            for (auto label : pair.second) {
                delete label;
            }
        }
        for (auto pair : m_downshift_labels) {
            for (auto label : pair.second) {
                delete label;
            }
        }

#ifndef SGX
        if (m_dev_crt_base != nullptr) {
            cudaCheckError(cudaFree(m_dev_crt_base));
        }
        if (m_dev_mrs_base != nullptr) {
            cudaCheckError(cudaFree(m_dev_mrs_base));
        }
        if (m_dev_zero_label != nullptr) {
            for (crt_val_t i = 2; i <= m_max_modulus; ++i) {
                cudaCheckError(cudaFree(m_dev_zero_label[i]));
            }
            cudaCheckError(cudaFree(m_dev_dev_zero_label));
            delete[] m_dev_zero_label;
        }
        if (m_dev_upshift_labels != nullptr) {
            for (size_t i = 0; i < m_crt_base.size(); ++i) {
                cudaCheckError(cudaFree(m_dev_upshift_labels[i]));
            }
            cudaCheckError(cudaFree(m_dev_dev_upshift_labels));
            delete[] m_dev_upshift_labels;
        }
        if (m_dev_downshift_labels != nullptr) {
            for (size_t i = 0; i < m_crt_base.size(); ++i) {
                cudaCheckError(cudaFree(m_dev_downshift_labels[i]));
            }
            cudaCheckError(cudaFree(m_dev_dev_downshift_labels));
            delete[] m_dev_downshift_labels;
        }
#else
        if (m_dev_crt_base != nullptr) {
            ocall_cudaFree(m_dev_crt_base);
        }
        if (m_dev_mrs_base != nullptr) {
            ocall_cudaFree(m_dev_mrs_base);
        }
        if (m_dev_zero_label != nullptr) {
            for (crt_val_t i = 2; i <= m_max_modulus; ++i) {
                ocall_cudaFree(m_dev_zero_label[i]);
            }
            ocall_cudaFree(m_dev_dev_zero_label);
            ocall_free(m_dev_zero_label);
        }
        if (m_dev_upshift_labels != nullptr) {
            for (size_t i = 0; i < m_crt_base.size(); ++i) {
                ocall_cudaFree(m_dev_upshift_labels[i]);
            }
            ocall_cudaFree(m_dev_dev_upshift_labels);
            ocall_free(m_dev_upshift_labels);
        }
        if (m_dev_downshift_labels != nullptr) {
            for (size_t i = 0; i < m_crt_base.size(); ++i) {
                ocall_cudaFree(m_dev_downshift_labels[i]);
            }
            ocall_cudaFree(m_dev_dev_downshift_labels);
            ocall_free(m_dev_downshift_labels);
        }
#endif
    }

    /**
     * @brief Encode raw values as labels for usage in garbled circuit.
     *
     * @param inputs
     * @return vector<LabelTensor*>*
     */
    vector<LabelTensor*>* garble_values(ScalarTensor<q_val_t>& values) {
        // reduce inputs to crt domain
        auto crt_values = util::crt_reduce<>(values, m_crt_base);

        // encode crt values as labels
        auto encoded_values = new vector<LabelTensor*>(m_crt_base.size());

        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            crt_val_t modul = m_crt_base.at(i);
            auto offset = get_label_offset(modul);
            auto label = new LabelTensor(offset, values.get_dims());
            *label *= crt_values.at(i);
            *label += get_zero_label(modul);
            encoded_values->at(i) = label;
        }
        return encoded_values;
    }

    /**
     * @brief Encode raw inputs as labels for input to a garbled circuit.
     *
     * @param inputs
     * @return vector<LabelTensor*>*
     */
    vector<LabelTensor*>* garble_inputs(const ScalarTensor<q_val_t>& inputs) {
        assert(m_circuit->get_input_dims() == inputs.get_dims() &&
               "Input dimension does not match circuit input dimension");
        // reduce inputs to crt domain
        auto crt_inputs = util::crt_reduce<>(inputs, m_crt_base);
        // encode crt inputs as labeltensors
        auto encoded_inputs = new vector<LabelTensor*>(m_crt_base.size());

        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            crt_val_t modul = m_crt_base.at(i);
            auto label_offset = get_label_offset(modul);
            auto label = new LabelTensor(label_offset, inputs.get_dims());
            *label *= crt_inputs.at(i);
            *label += *m_base_label.at(i);
            encoded_inputs->at(i) = label;
        }

        return encoded_inputs;
    }

    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads = -1) {
        auto encoded_inputs_tmp{encoded_inputs};
        auto layer = m_circuit->get_layer();
        for (size_t i = 0; i < layer.size(); ++i) {
            // Handle flatten layer separately
            if (layer.at(i)->get_type() == Layer::flatten) {
                for (auto input : *encoded_inputs_tmp) {
                    input->flatten();
                }
                if (i == layer.size()) return encoded_inputs_tmp;
            }
            if (i == m_garbled_layer.size()) return encoded_inputs_tmp;

            int threads = nr_threads;
            if (threads == -1) threads = DEFAULT_NUM_THREADS;

#ifdef BENCHMARK
            auto t1 = high_resolution_clock::now();
#endif
            encoded_inputs_tmp = m_garbled_layer.at(i)->cpu_evaluate(
                encoded_inputs_tmp, threads);
#ifdef BENCHMARK
            auto t2 = high_resolution_clock::now();
            auto g_layer_type = m_garbled_layer.at(i)->get_layer()->get_type();
            auto name_string = m_layer_names.at(g_layer_type);
            duration<double, std::milli> ms_double = t2 - t1;
            if (m_eval_times.find(name_string) == m_eval_times.end()) {
                m_eval_times[name_string] = ms_double.count();
            } else {
                m_eval_times[name_string] += ms_double.count();
            }
#endif
        }
        return encoded_inputs_tmp;
    }

    /**
     * @brief Generates decoding information for the garbled circuit.
     *
     */
    void gen_decoding_information() {
        m_decoding_information.resize(m_crt_base.size());

        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            crt_val_t modul = m_crt_base.at(i);
            m_decoding_information.at(i).reserve(modul);
            for (crt_val_t k = 0; k < modul; ++k) {
                LabelTensor label;
                if (m_garbled_layer.size() > 0)
                    label = *m_garbled_layer.back()->get_out_label()->at(i);
                else {
                    label = *m_base_label.at(i);
                }
                // LabelTensor label{modul, 0, m_circuit->get_output_dims()};
                label += {k * get_label_offset(modul)};
                label.compress();

                // m_garbled_layer.back()->get_out_label()->at(i)->compress();
                // for(int j = 0; j < m_circuit->get_output_size(); ++j) {
                //     __uint128_t tmp = label.get_compressed()[j] +
                //     m_garbled_layer.back()->get_out_label()->at(i)->get_compressed()[j];
                //     label.set_compressed(tmp, j);
                // }
                label.hash();
                ScalarTensor<__uint128_t> modul_dec_inf{label.get_hashed(),
                                                        label.get_dims()};
                m_decoding_information.at(i).push_back(modul_dec_inf);
            }
        }
    }

    /**
     * @brief Decode given garbled outputs with dec. inf. of this gc
     *
     * @return vector<q_val_t>
     */
    ScalarTensor<q_val_t> decode_outputs(vector<LabelTensor*>* g_output) {
        vector<ScalarTensor<crt_val_t>> crt_value;
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            crt_value.push_back(ScalarTensor<crt_val_t>{});
            crt_value.back().resize(m_circuit->get_output_dims());
        }

        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            // std::cerr << "decoding for plain modulus: " << m_crt_base.at(i) << std::endl;
            size_t found = 0;
            for (int j = 0; j < m_crt_base.at(i); ++j) {
                LabelTensor dec{*g_output->at(i)};
                dec.compress();
                dec.hash();
                for (size_t l = 0; l < m_circuit->get_output_size(); ++l) {
                    if (dec.get_hashed()[l] ==
                        m_decoding_information.at(i).at(j).data()[l]) {
                        crt_value.at(i).data()[l] = j;
                        found++;
                        // print which one was found
                        // std::cerr << "decoded plain value " << j << " for output " << l << std::endl;
                    }
                }
                if (found == m_circuit->get_output_size()) break;
            }
            if (found != m_circuit->get_output_size()) {
                std::cerr << "ERROR: mismatch in decoding for plain modulus: " << m_crt_base.at(i) << std::endl;
                std::cerr << "found: " << found << std::endl;
                std::cerr << "m_circuit->get_output_size(): " << m_circuit->get_output_size() << std::endl;
            }
            assert(found == m_circuit->get_output_size() &&
                   "Decoding failed, no matching label found");
        }

        auto result =
            util::chinese_remainder<crt_val_t, q_val_t>(m_crt_base, crt_value);

        // Map outputs back to signed domain
        for (size_t i = 0; i < result.size(); ++i) {
            if (result.data()[i] >= m_crt_modulus / 2) {
                result.data()[i] -= m_crt_modulus;
            }
        }

        return result;
    }

    /**
     * @brief Helper routine to garble circuit.
     *
     */
    virtual void garble() = 0;

#ifndef SGX

    void cuda_move() {
        // Initialize cuda_aes_engine
        CUDAAESEngine cuda_aes_engine{};

        // Move crt base
        cudaCheckError(cudaMalloc(reinterpret_cast<void**>(&m_dev_crt_base),
                                  m_crt_base.size() * sizeof(crt_val_t)));
        cudaCheckError(cudaMemcpy(m_dev_crt_base, m_crt_base.data(),
                                  m_crt_base.size() * sizeof(crt_val_t),
                                  cudaMemcpyHostToDevice));

        // Move mrs base
        cudaCheckError(cudaMalloc(reinterpret_cast<void**>(&m_dev_mrs_base),
                                  m_mrs_base.size() * sizeof(mrs_val_t)));
        cudaCheckError(cudaMemcpy(m_dev_mrs_base, m_mrs_base.data(),
                                  m_mrs_base.size() * sizeof(mrs_val_t),
                                  cudaMemcpyHostToDevice));

        // Move zero label
        int size = (m_max_modulus + 1) * sizeof(crt_val_t*);
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_dev_zero_label), size));
        m_dev_zero_label = new crt_val_t*[m_max_modulus + 1];
        for (crt_val_t i = 2; i <= m_max_modulus; ++i) {
            size_t size = get_zero_label(i).get_nr_comps() * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_zero_label[i]), size));
            cudaCheckError(cudaMemcpy(m_dev_zero_label[i],
                                      get_zero_label(i).get_components(), size,
                                      cudaMemcpyHostToDevice));
        }
        cudaCheckError(cudaMemcpy(m_dev_dev_zero_label, m_dev_zero_label, size,
                                  cudaMemcpyHostToDevice));

        // Move static input labels for garbled rescaling
        //// Upschift labels
        size = m_crt_base.size() * sizeof(crt_val_t*);
        cudaCheckError(cudaMalloc(
            reinterpret_cast<void**>(&m_dev_dev_upshift_labels), size));
        m_dev_upshift_labels = new crt_val_t*[m_crt_base.size()];
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            size_t size = m_upshift_labels.at(i)->size() * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_upshift_labels[i]), size));
            cudaCheckError(cudaMemcpy(m_dev_upshift_labels[i],
                                      get_upshift_label(i).get_components(),
                                      size, cudaMemcpyHostToDevice));
        }
        cudaCheckError(cudaMemcpy(m_dev_dev_upshift_labels,
                                  m_dev_upshift_labels, size,
                                  cudaMemcpyHostToDevice));

        //// Downshift labels
        cudaCheckError(cudaMalloc(
            reinterpret_cast<void**>(&m_dev_dev_downshift_labels), size));
        m_dev_downshift_labels = new crt_val_t*[m_crt_base.size()];
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            size_t size = get_downshift_label({2}, i).size() * sizeof(crt_val_t); // FM 30.10.24: previously, m_downshift_labels contained only one element (for s = 2 = base.at(0))
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_downshift_labels[i]), size));
            cudaCheckError(cudaMemcpy(m_dev_downshift_labels[i],
                                      get_downshift_label({2}, i).get_components(), // FM 30.10.24: previously, m_downshift_labels contained only one element (for s = 2 = base.at(0))
                                      size, cudaMemcpyHostToDevice));
        }
        cudaCheckError(cudaMemcpy(m_dev_dev_downshift_labels,
                                  m_dev_downshift_labels, size,
                                  cudaMemcpyHostToDevice));

        // move layer by layer to device
        for (auto layer : m_garbled_layer) {
            layer->cuda_move();
        }
    }

    crt_val_t** cuda_move_inputs(vector<LabelTensor*>* g_inputs) {
        // allocate pointer array
        crt_val_t** dev_garbled_inputs = new crt_val_t*[m_crt_base.size()];

        // allocate device memory and copy data to device
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            size_t size = g_inputs->at(i)->size() * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&dev_garbled_inputs[i]), size));
            cudaCheckError(cudaMemcpy(dev_garbled_inputs[i],
                                      g_inputs->at(i)->get_components(), size,
                                      cudaMemcpyHostToDevice));
        }
        return dev_garbled_inputs;
    }

    /**
     * @brief Free garbled inputs on device
     *
     * @param g_inputs
     */
    void cuda_free_inputs(crt_val_t** dev_g_inputs) {
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            cudaCheckError(cudaFree(dev_g_inputs[i]));
        }
        delete[] dev_g_inputs;
    }
#ifdef BENCHMARK
    void cuda_evaluate(crt_val_t** dev_g_inputs) {
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
        for (size_t i = 0; i < m_garbled_layer.size(); ++i) {
            if (i > 0) {
                t1 = std::chrono::high_resolution_clock::now();
                m_garbled_layer.at(i)->cuda_evaluate(
                    m_garbled_layer.at(i - 1)->get_dev_out_label());
                t2 = std::chrono::high_resolution_clock::now();
            } else {
                t1 = std::chrono::high_resolution_clock::now();
                m_garbled_layer.at(i)->cuda_evaluate(dev_g_inputs);
                t2 = std::chrono::high_resolution_clock::now();
            }
            auto layer_type = m_garbled_layer.at(i)->get_layer()->get_type();
            auto name_string = m_layer_names.at(layer_type);
            duration<double, std::milli> ms_double = t2 - t1;
            if (m_eval_times.find(name_string) == m_eval_times.end()) {
                m_eval_times[name_string] = ms_double.count();
            } else {
                m_eval_times[name_string] += ms_double.count();
            }
        }
    }
#else
    void cuda_evaluate(crt_val_t** dev_g_inputs) {
        for (size_t i = 0; i < m_garbled_layer.size(); ++i) {
            if (i > 0) {
                m_garbled_layer.at(i)->cuda_evaluate(
                    m_garbled_layer.at(i - 1)->get_dev_out_label());
            } else {
                m_garbled_layer.at(i)->cuda_evaluate(dev_g_inputs);
            }
        }
    }
#endif  // Benchmark

#else

    void cuda_move() {
        ocall_init_cuda_aes_engine();

        // Move crt base
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_crt_base),
                         m_crt_base.size() * sizeof(crt_val_t));
        sgx_cudaMemcpyToDevice(m_dev_crt_base, m_crt_base.data(),
                               m_crt_base.size() * sizeof(crt_val_t));

        // Move mrs base
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_mrs_base),
                         m_mrs_base.size() * sizeof(mrs_val_t));
        sgx_cudaMemcpyToDevice(m_dev_mrs_base, m_mrs_base.data(),
                               m_mrs_base.size() * sizeof(mrs_val_t));

        // Move zero label
        int size = (m_max_modulus + 1) * sizeof(crt_val_t*);
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_dev_zero_label), size);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_zero_label),
                              size);
        for (crt_val_t i = 2; i <= m_max_modulus; ++i) {
            int size = get_zero_label(i).get_nr_comps() * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_zero_label[i]),
                             size);
            sgx_cudaMemcpyToDevice(m_dev_zero_label[i],
                                   get_zero_label(i).get_components(), size);
        }
        ocall_cudaMemcpyToDevicePtr(
            reinterpret_cast<void**>(m_dev_dev_zero_label),
            reinterpret_cast<void**>(m_dev_zero_label), size);

        // Move static input labels for garbled rescaling
        //// Upschift labels
        size = m_crt_base.size() * sizeof(crt_val_t*);
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_dev_upshift_labels),
                         size);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_upshift_labels),
                              size);
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            int size = m_upshift_labels.at(i)->size() * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_upshift_labels[i]),
                             size);
            sgx_cudaMemcpyToDevice(m_dev_upshift_labels[i],
                                   get_upshift_label(i).get_components(), size);
        }
        ocall_cudaMemcpyToDevicePtr(
            reinterpret_cast<void**>(m_dev_dev_upshift_labels),
            reinterpret_cast<void**>(m_dev_upshift_labels), size);

        //// Downshift labels
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_dev_downshift_labels),
                         size);
        ocall_alloc_ptr_array(
            reinterpret_cast<void***>(&m_dev_downshift_labels), size);
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            int size = get_downshift_label({2}, i).size() * sizeof(crt_val_t); // FM 30.10.24: previously, m_downshift_labels contained only one element (for s = 2 = base.at(0))
            ocall_cudaMalloc(
                reinterpret_cast<void**>(&m_dev_downshift_labels[i]), size);
            sgx_cudaMemcpyToDevice(m_dev_downshift_labels[i],
                                   get_downshift_label({2}, i).get_components(),
                                   size);
        }
        ocall_cudaMemcpyToDevicePtr(
            reinterpret_cast<void**>(m_dev_dev_downshift_labels),
            reinterpret_cast<void**>(m_dev_downshift_labels), size);

        // move layer by layer to device
        for (auto layer : m_garbled_layer) {
            layer->cuda_move();
        }
    }

    crt_val_t** cuda_move_inputs(vector<LabelTensor*>* g_inputs) {
        // allocate pointer array
        int size = m_crt_base.size() * sizeof(crt_val_t*);
        crt_val_t** dev_garbled_inputs;
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&dev_garbled_inputs),
                              size);

        // allocate device memory and copy data to device
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            int size = g_inputs->at(i)->size() * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&dev_garbled_inputs[i]),
                             size);
            sgx_cudaMemcpyToDevice(dev_garbled_inputs[i],
                                   g_inputs->at(i)->get_components(), size);
        }

        return dev_garbled_inputs;
    }

    /**
     * @brief Free garbled inputs on device
     *
     * @param g_inputs
     */
    void cuda_free_inputs(crt_val_t** dev_g_inputs) {
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            ocall_cudaFree(dev_g_inputs[i]);
        }
        ocall_free(dev_g_inputs);
    }

    void cuda_evaluate(crt_val_t** dev_g_inputs) {
        for (size_t i = 0; i < m_garbled_layer.size(); ++i) {
            if (i > 0) {
                m_garbled_layer.at(i)->cuda_evaluate(
                    m_garbled_layer.at(i - 1)->get_dev_out_label());
            } else {
                m_garbled_layer.at(i)->cuda_evaluate(dev_g_inputs);
            }
        }
    }

#endif

    vector<LabelTensor*>* cuda_move_outputs() {
        m_garbled_layer.back()->cuda_move_output();
        return m_garbled_layer.back()->get_out_label();
    }

    crt_val_t** get_dev_dev_zero_label() { return m_dev_dev_zero_label; }
    crt_val_t* get_dev_zero_label(crt_val_t modulus) {
        return m_dev_zero_label[modulus];
    }
    crt_val_t** get_dev_zero_label() { return m_dev_zero_label; }
    crt_val_t* get_dev_crt_base() { return m_dev_crt_base; }
    mrs_val_t* get_dev_mrs_base() { return m_dev_mrs_base; }

    crt_val_t** get_dev_upshift_labels() { return m_dev_upshift_labels; }
    crt_val_t** get_dev_downshift_labels() { return m_dev_downshift_labels; }

    crt_val_t** get_dev_dev_upshift_labels() {
        return m_dev_dev_upshift_labels;
    }
    crt_val_t** get_dev_dev_downshift_labels() {
        return m_dev_dev_downshift_labels;
    }

    Circuit* get_circuit() const { return m_circuit; }
    vector<crt_val_t>& get_crt_base() { return m_crt_base; }
    q_val_t get_crt_modulus() const { return m_crt_modulus; }
    vector<mrs_val_t>& get_mrs_base() { return m_mrs_base; }
    vector<GarbledLayer*>& get_garbled_layer() { return m_garbled_layer; }
    vector<vector<ScalarTensor<__uint128_t>>>& get_decoding_information() {
        return m_decoding_information;
    }
    LabelSlice get_zero_label(int modulus) {
        assert(modulus < m_zero_label.size() + 2 &&
               "Zero-label not available, set max modulus of garbled circuit!");
        return m_zero_label.at(modulus - 2)->get_label(0);
    }
    vector<LabelTensor*>* get_zero_labels() { return &m_zero_label; }
    LabelSlice get_label_offset(int modulus) {
        assert(
            modulus < m_label_offset.size() + 2 &&
            "Offset-label not available, set max modulus of garbled circuit!");
        return m_label_offset.at(modulus - 2)->get_label(0);
    }
    vector<LabelTensor*>& get_base_label() { return m_base_label; }
    crt_val_t get_max_modulus() const { return m_max_modulus; }
    LookupApproxSign* get_lookup_approx_sign() { return m_lookup_approx_sign; }

    LabelTensor get_upshift_label_base(size_t modulus_idx) {
        return *m_upshift_labels_base.at(modulus_idx);
    }

    LabelTensor get_upshift_label(size_t modulus_idx) {
        return *m_upshift_labels.at(modulus_idx);
    }

    LabelTensor get_downshift_label_base(const vector<crt_val_t> scaling_factors, size_t modulus_idx) {
        const auto scaling_factors_prod = std::accumulate(scaling_factors.begin(), scaling_factors.end(), 1, std::multiplies<crt_val_t>());

        if (m_downshift_labels_base.find(scaling_factors_prod) != m_downshift_labels_base.end()) {
            return *m_downshift_labels_base.at(scaling_factors_prod).at(modulus_idx);
        }
        else {
            const auto labels = create_downshift_base_labels();
            m_downshift_labels_base[scaling_factors_prod] = labels;
            return *labels.at(modulus_idx);
        }
    }

    LabelTensor get_downshift_label(const vector<crt_val_t> scaling_factors, size_t modulus_idx) {
        const auto scaling_factors_prod = std::accumulate(scaling_factors.begin(), scaling_factors.end(), 1, std::multiplies<crt_val_t>());

        if (m_downshift_labels.find(scaling_factors_prod) != m_downshift_labels.end()) {
            return *m_downshift_labels.at(scaling_factors_prod).at(modulus_idx);
        }
        else {
            const auto labels = create_downshift_labels(scaling_factors);
            m_downshift_labels[scaling_factors_prod] = labels;
            return *labels.at(modulus_idx);
        }
    }

#ifdef BENCHMARK
    std::map<std::string, double> get_evaluation_times() {
        return m_eval_times;
    }
#endif

   private:
    inline void init_random_label(vector<LabelTensor*>& labels,
                                  bool offset_value = false) {
        labels.reserve(m_max_modulus - 1);
        for (crt_val_t modulus = 2; modulus <= m_max_modulus; ++modulus) {
            // insert at index modulus to enable indexing with modulus
            auto label = new LabelTensor(modulus, dim_t{1});
            label->init_random();
            if (offset_value) label->set_offset_value();
            labels.push_back(label);
        }
    }

    inline void init_base_label(dim_t circuit_input_dims) {
        for (auto base : m_crt_base) {
            auto label = new LabelTensor(base, circuit_input_dims);
            label->init_random();
            m_base_label.push_back(label);
        }
    }

    /**
     * @brief Compute the maximal modulus needed for the garbled circuit.
     *
     * If the approx. sign computation is part of the circuit, the maximal
     * modulus is defined by the third step of the mrs addition.
     *
     * @param crt_base
     * @param mrs_base
     * @return q_val_t
     */
    inline crt_val_t compute_max_modulus(const vector<crt_val_t>& crt_base,
                                         const vector<mrs_val_t>& mrs_base) {
        // at(1), because the most significant value of the mrs base is not
        // considered in the mrs addition of the approx. sign computation
        // + 1 for the carry value in the mrs addition
        crt_val_t mrs_induced_max = 0;
        if (mrs_base.size() > 1)
            mrs_induced_max = (crt_base.size() + 1) * mrs_base.at(1);
        // approx_rex_gadget needs minimal modulus max(mrs_base)
        mrs_induced_max = std::max(mrs_induced_max, mrs_base.at(0));
        crt_val_t crt_induced_max = crt_base.at(0);
        crt_val_t max_modulus = std::max(mrs_induced_max, crt_induced_max);
        return max_modulus;
    }

    // Lookup for mrs base depending on the used crt_base and desired approx.
    // sign accuracy
    //// Hashfunction needed for unordered map
    // https://stackoverflow.com/a/32685618/8538713
    // Only for pairs of std::hash-able types for simplicity.
    // You can of course template this struct to allow other hash functions
    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);

            // Mainly for demonstration purposes, i.e. works but is overly
            // simple In the real world, use sth. like boost.hash_combine
            return h1 ^ h2;
        }
    };
    //// key value of unordered map is pair of crt_base and approx. sign acc.
    typedef std::pair<int, float> key_t;
    //// value of unordered map is mrs_base
    typedef std::vector<mrs_val_t> value_t;

    static value_t get_mrs_base(int crt_base_len, float approx_sign_accuracy) {
        static std::unordered_map<key_t, value_t, pair_hash> lookup{
            // k = 4
            {{4, 100.0}, {26, 3}},
            {{4, 99.0}, {18, 3}},
            // k = 5
            {{5, 100.0}, {54, 4, 3}},
            {{5, 99.9}, {30, 5, 3}},
            {{5, 99.0}, {36, 3}},
            // k = 6
            {{6, 100.0}, {60, 5, 5, 5}},
            {{6, 99.99}, {42, 5, 5, 5}},
            {{6, 99.9}, {48, 5, 4}},
            {{6, 99.0}, {40, 3}},
            // k = 7
            {{7, 100.0}, {86, 7, 6, 6, 5}},
            {{7, 99.99}, {88, 6, 5, 4}},
            {{7, 99.9}, {60, 5, 4}},
            {{7, 99.0}, {40, 3}},
            // k = 8
            {{8, 100.0}, {98, 9, 8, 8, 7, 5}},
            {{8, 99.999}, {102, 7, 6, 5, 5}},
            {{8, 99.99}, {78, 7, 5, 4}},
            {{8, 99.9}, {78, 5, 3}},
            {{8, 99.0}, {126}},
            // k = 9
            {{9, 100.0}, {76, 7, 7, 7, 7, 7, 5, 5}},
            {{9, 99.999}, {114, 7, 6, 5, 5}},
            {{9, 99.99}, {84, 6, 5, 5}},
            {{9, 99.9}, {140, 9}},
            {{9, 99.0}, {138}},
            // k = 10
            {{10, 100.0}, {202, 11, 11, 6, 6, 6, 6, 5, 5}},
            {{10, 99.999}, {102, 7, 6, 6, 5}},
            {{10, 99.99}, {112, 6, 5, 4}},
            {{10, 99.9}, {190, 7}},
            {{10, 99.0}, {140}},
            // k = 11
            {{11, 100}, {150, 8, 7, 7, 6, 6, 6, 5, 5, 5, 5, 5}},
            {{11, 99.999}, {130, 7, 6, 5, 5}},
            {{11, 99.99}, {174, 11, 7}}};
        return lookup.at({crt_base_len, approx_sign_accuracy});
    }

   protected:
    void init_garbling() {
        init_random_label(m_zero_label);
        init_random_label(m_label_offset, true);
        init_base_label(m_circuit->get_layer()[0]->get_input_dims());

        // Init constant labels for garbled rescaling
        //// Upshift base labels
        for (auto base : m_crt_base) {
            auto label = new LabelTensor{base};
            label->init_random();
            m_upshift_labels_base.push_back(label);
        }

        //// Upshift labels
        m_upshift_labels.resize(m_crt_base.size());
        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            m_upshift_labels.at(i) =
                new LabelTensor{get_label_offset(m_crt_base.at(i))};
            *m_upshift_labels.at(i) *=
                util::modulo(m_crt_modulus / 2, m_crt_base.at(i));
            *m_upshift_labels.at(i) += *m_upshift_labels_base.at(i);
        }
    }

    //TODO: unify these 2 in style
    vector<LabelTensor * > create_downshift_base_labels() {
        auto labels = vector<LabelTensor * >{};

        for (auto base : m_crt_base) {
            auto label = new LabelTensor{base};
            label->init_random();
            labels.push_back(label);
        }

        return labels;
    }

    vector<LabelTensor *> create_downshift_labels(const vector<crt_val_t> scaling_factors) {
        const auto prod_scaling_factors = std::accumulate(scaling_factors.begin(), scaling_factors.end(), 1, std::multiplies<crt_val_t>());
        auto labels = vector<LabelTensor * >{m_crt_base.size(), nullptr};

        for (size_t i = 0; i < m_crt_base.size(); ++i) {
            labels.at(i) = new LabelTensor{get_label_offset(m_crt_base.at(i))};
            *labels.at(i) *= util::modulo(m_crt_modulus / (2 * prod_scaling_factors), m_crt_base.at(i));
            *labels.at(i) += get_downshift_label_base(scaling_factors, i);
        }

        return labels;
    }
};

#endif
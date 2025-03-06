#ifndef RESCALE_GADGET_H
#define RESCALE_GADGET_H

#include <vector>

#include "garbling/gadgets/sign_gadget.h"
#include "garbling/gadgets/base_extension_gadget.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/projection_gate.h"
#include "misc/datatypes.h"
#include "misc/util.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "misc/cuda_util.h"
#endif

using std::vector;

class RescaleGadget
{
    GarbledCircuitInterface *m_gc;
    size_t m_input_size;
    const bool m_use_sign_base_extension;
    const vector<crt_val_t> m_scaling_factors;
    vector<size_t> m_scaling_factors_indices;

    // Step 2.1: Ciphertexts needed to project modulus 2 label
    //// factor_count * crt_base_size * input_size
    vector<vector<vector<ProjectionGate *>>> m_proj_gates_trans_mod;
    //// size: input_size * (crt_base_size - 1) * total scaling factor
    __uint128_t *m_trans_mod_ciphers;
    // Step 3: Base Extension
    SignGadget *m_sign_gadget{nullptr};
    BaseExtensionGadget *m_be_gadget{nullptr};

    crt_val_t **m_dev_out_label{nullptr};
    __uint128_t *m_dev_trans_mod_ciphers{nullptr};

public:
    RescaleGadget(GarbledCircuitInterface *gc, size_t input_size, const bool use_sign_base_extension, const vector<crt_val_t> scaling_factors)
        : m_gc{gc},
          m_input_size{input_size},
          m_use_sign_base_extension{use_sign_base_extension},
          m_scaling_factors{scaling_factors}
    {
        for (auto factor : m_scaling_factors)
        {
            const auto factor_index = std::find(m_gc->get_crt_base().begin(), m_gc->get_crt_base().end(), factor) - m_gc->get_crt_base().begin();
            m_scaling_factors_indices.push_back(factor_index);
        }

        // Reserve memory for projection gates
        int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                           m_gc->get_crt_base().end(), 0);
        size_t crt_base_size = m_gc->get_crt_base().size();

        // Step 2.1 (and step 1): Ciphertexts needed to project modulus m_scaling_factors labels
        const auto scaling_factors_prod = std::accumulate(m_scaling_factors.begin(), m_scaling_factors.end(), 1, std::multiplies<crt_val_t>());
        size_t size = m_input_size * (crt_base_size - 1) * scaling_factors_prod;
        m_trans_mod_ciphers = new __uint128_t[size];

        // debug_print_all_constants();
    }

    ~RescaleGadget()
    {
        for (auto &projs : m_proj_gates_trans_mod)
        {
            for (auto &proj : projs)
            {
                for (auto &gate : proj)
                {
                    delete gate;
                }
            }
        }
        if (m_sign_gadget != nullptr)
        {
            delete m_sign_gadget;
        }
        if (m_be_gadget != nullptr)
        {
            delete m_be_gadget;
        }

        delete[] m_trans_mod_ciphers;

#ifndef SGX
        if (m_dev_trans_mod_ciphers != nullptr)
            cudaCheckError(cudaFree(m_dev_trans_mod_ciphers));
        if (m_dev_out_label != nullptr)
        {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i)
            {
                cudaCheckError(cudaFree(m_dev_out_label[i]));
            }
            delete[] m_dev_out_label;
        }
#else
        if (m_dev_trans_mod_ciphers != nullptr)
            ocall_cudaFree(m_dev_trans_mod_ciphers);
        if (m_dev_out_label != nullptr)
        {
            for (size_t i = 0; i < m_gc->get_crt_base().size(); ++i)
            {
                ocall_cudaFree(m_dev_out_label[i]);
            }
            ocall_free(m_dev_out_label);
        }
#endif
    }

    void garble(vector<LabelTensor *> *in_label,
                vector<LabelTensor *> *out_label)
    {
        size_t crt_base_size = m_gc->get_crt_base().size();
        size_t factor_count = m_scaling_factors.size();
        dim_t output_dims = in_label->at(0)->get_dims();
        vector<size_t> ignored_modulus_indices = {};

        m_proj_gates_trans_mod = vector<vector<vector<ProjectionGate *>>>{
            factor_count,
            vector<vector<ProjectionGate *>>{
                crt_base_size,
                vector<ProjectionGate *>(m_input_size, nullptr)}};
        size_t proj_cipher_cnt = 0;

        // Step 1: Upshift by max_crt_modulus / 2
        for (size_t i = 0; i < m_input_size; ++i)
        {
            for (size_t j = 0; j < crt_base_size; ++j)
            {
                crt_val_t modulus = m_gc->get_crt_base().at(j);
                LabelTensor input_label{in_label->at(j)->get_label(i)};
                input_label += m_gc->get_upshift_label_base(j);
                out_label->at(j)->set_label(input_label.get_label(0), i);
            }
        }

        // Step 2: Rescale
        for (size_t i = 0; i < factor_count; ++i)
        {
            const auto factor_idx = std::find(m_gc->get_crt_base().begin(), m_gc->get_crt_base().end(), m_scaling_factors[i]) - m_gc->get_crt_base().begin();
            ignored_modulus_indices.push_back(factor_idx);

            for (size_t j = 0; j < crt_base_size; ++j)
            {
                // Check if modulus is ignored
                if (std::find(ignored_modulus_indices.begin(), ignored_modulus_indices.end(), j) == ignored_modulus_indices.end())
                {
                    for (size_t k = 0; k < m_input_size; ++k)
                    {
                        // Step 2.1: Subtraction
                        // a) project residue modulus scaling factor to modulus crt_base.at(j)
                        // b) subtract this value from the input label at modulus j...
                        crt_val_t modulus = m_gc->get_crt_base().at(j);
                        auto out_offset_label = m_gc->get_label_offset(modulus);
                        auto input_label = LabelTensor{out_label->at(j)->get_label(k)};

                        //// a) Projection
                        auto id = &ProjectionFunctionalities::identity;
                        auto proj_gate = new ProjectionGate{modulus, &m_trans_mod_ciphers[proj_cipher_cnt], id, m_gc};
                        m_proj_gates_trans_mod.at(i).at(j).at(k) = proj_gate;
                        proj_cipher_cnt += m_scaling_factors[i];

                        LabelTensor out_base_label{modulus};
                        out_base_label.init_random();
                        auto mod_ext_label = out_label->at(factor_idx)->get_label(k);
                        proj_gate->garble(mod_ext_label, &out_base_label, out_offset_label);

                        //// b) Subtraction
                        input_label -= out_base_label;

                        // Step 2.2: Inverse multiplication
                        crt_val_t inverse_mod = util::mul_inv(m_scaling_factors[i], modulus);
                        input_label *= inverse_mod;

                        out_label->at(j)->set_label(input_label.get_label(0), k);
                    }
                }
            }
        }

        // Step 2.3: Fill in zeroes for the ignored moduli
        for (const auto ignored_idx : ignored_modulus_indices)
        {
#ifndef SGX
#pragma omp parallel for
#endif
            for (size_t i = 0; i < m_input_size; ++i)
            {
                const auto factor = m_gc->get_crt_base().at(ignored_idx);
                out_label->at(ignored_idx)->set_label(m_gc->get_zero_label(factor), i);
            }
        }

        // Step 3: Base Extension
        m_be_gadget = new BaseExtensionGadget{m_gc, m_input_size, m_gc->get_crt_base(), m_scaling_factors};
        m_be_gadget->garble(out_label, out_label);

        // Step 4: Downshift by max_crt_modulus / (2 * scaling_factor)
        for (size_t i = 0; i < m_input_size; ++i)
        {
            for (size_t j = 0; j < crt_base_size; ++j)
            {
                crt_val_t modulus = m_gc->get_crt_base().at(j);
                LabelTensor input_label{out_label->at(j)->get_label(i)};
                input_label -= m_gc->get_downshift_label_base(m_scaling_factors, j);
                out_label->at(j)->set_label(input_label.get_label(0), i);
            }
        }
    }

    void cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                      vector<LabelTensor *> *out_label, int nr_threads)
    {
        size_t crt_base_size = m_gc->get_crt_base().size();
        size_t factor_count = m_scaling_factors.size();
        dim_t output_dims = encoded_inputs->at(0)->get_dims();
        vector<size_t> ignored_modulus_indices = {};

        // Step 1: Upshift by max_crt_modulus / 2
        for (size_t i = 0; i < m_input_size; ++i)
        {
            for (size_t j = 0; j < crt_base_size; ++j)
            {
                crt_val_t modulus = m_gc->get_crt_base().at(j);
                LabelTensor input_label{encoded_inputs->at(j)->get_label(i)};
                input_label += m_gc->get_upshift_label(j);
                out_label->at(j)->set_label(input_label.get_label(0), i);
            }
        }

        // Step 2: Rescale
        for (size_t i = 0; i < factor_count; ++i)
        {
            const auto factor_idx = std::find(m_gc->get_crt_base().begin(), m_gc->get_crt_base().end(), m_scaling_factors[i]) - m_gc->get_crt_base().begin();
            ignored_modulus_indices.push_back(factor_idx);

            for (size_t j = 0; j < crt_base_size; ++j)
            {
                // Check if modulus is ignored
                if (std::find(ignored_modulus_indices.begin(), ignored_modulus_indices.end(), j) == ignored_modulus_indices.end())
                {
#ifndef SGX
#pragma omp parallel for
#endif
                    for (size_t k = 0; k < m_input_size; ++k)
                    {
                        // Step 2.1: Subtraction
                        // a) project residue modulus scaling factor to modulus crt_base.at(j)
                        // b) subtract this value from the input label at modulus j...

                        //// a) Projection
                        auto proj = m_proj_gates_trans_mod.at(i).at(j).at(k);
                        auto modulus = m_gc->get_crt_base().at(j);
                        auto input_label = LabelTensor{out_label->at(j)->get_label(k)};
                        auto mod_ext_label = out_label->at(factor_idx)->get_label(k);
                        auto step_out_label = proj->cpu_evaluate(mod_ext_label);
                        //// b) Subtraction
                        input_label -= step_out_label;

                        // Step 2.2: Inverse multiplication
                        crt_val_t inverse_mod = util::mul_inv(m_scaling_factors[i], modulus);
                        input_label *= inverse_mod;

                        out_label->at(j)->set_label(input_label.get_label(0), k);
                    }
                }
            }
        }

        // Step 2.3: Fill in zeroes for the ignored moduli
        for (const auto ignored_idx : ignored_modulus_indices)
        {
#ifndef SGX
#pragma omp parallel for
#endif
            for (size_t i = 0; i < m_input_size; ++i)
            {
                const auto factor = m_gc->get_crt_base().at(ignored_idx);
                out_label->at(ignored_idx)->set_label(m_gc->get_zero_label(factor), i);
            }
        }

        // Step 3: Base Extension
        m_be_gadget->cpu_evaluate(out_label, out_label, nr_threads);

        // Step 4: Downshift by max_crt_modulus / (2 * scaling_factor)
        for (size_t i = 0; i < m_input_size; ++i)
        {
            for (size_t j = 0; j < crt_base_size; ++j)
            {
                crt_val_t modulus = m_gc->get_crt_base().at(j);
                LabelTensor input_label{out_label->at(j)->get_label(i)};
                input_label -= m_gc->get_downshift_label(m_scaling_factors, j);
                out_label->at(j)->set_label(input_label.get_label(0), i);
            }
        }
    }

// TODO: CUDA doesnt work currently
#ifndef SGX
    void cuda_move()
    {
        // Allocate memory fo the output labels
        //// Allocate pointer array
        m_dev_out_label = new crt_val_t *[m_gc->get_crt_base().size()];
        //// Allocate device array
        size_t output_size = m_input_size;
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t size = nr_comps * output_size * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc((void **)&m_dev_out_label[i], size));
        }

        // Move ciphertexts of proj_gates_trans_mod
        size_t size = m_input_size * (crt_base_size - 1) * 2;
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void **>(&m_dev_trans_mod_ciphers),
                       size * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_trans_mod_ciphers, m_trans_mod_ciphers,
                                  size * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));

        // Move sign gadget
        m_sign_gadget->cuda_move();
    }

    crt_val_t **cuda_evaluate(crt_val_t **dev_in_label)
    {
        size_t crt_base_size = m_gc->get_crt_base().size();
        cudaStream_t stream[crt_base_size];

        size_t nr_blocks = ceil_div(m_input_size, 32lu);

        // Step 1: Upshift by max_crt_modulus / 2
        crt_val_t **dev_upshift_labels = m_gc->get_dev_upshift_labels();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            cudaStreamCreate(&stream[i]);

            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);

            AddLabel<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_out_label[i], dev_in_label[i], dev_upshift_labels[i],
                modulus, m_input_size, nr_comps);
        }
        cudaDeviceSynchronize();

        // Step 2.1
        // a) project residue modulus 2 to modulus crt_base.at(j)
        // b) subtract this value from the input label at
        // modulus crt_base.at(j)...
        // c) multiply by inverse of 2 mod crt_base.at(j)
        for (size_t i = 1; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            crt_val_t inverse_mod = util::mul_inv(2, modulus);
            Scale_1<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_out_label[i], m_dev_out_label[i], m_dev_out_label[0],
                m_dev_trans_mod_ciphers, modulus, m_input_size, i,
                crt_base_size, inverse_mod);
        }
        cudaDeviceSynchronize();

        // Step 3: Base Extension
        //// Set first result to zero
        size_t nr_components = LabelTensor::get_nr_comps(2);
        crt_val_t *dev_zero_label = m_gc->get_dev_zero_label(2);
        DevDevCopy<<<nr_blocks, 32>>>(m_dev_out_label[0], dev_zero_label,
                                      m_input_size, nr_components);
        cudaDeviceSynchronize();
        //// Sign gadget
        m_sign_gadget->cuda_evaluate(m_dev_out_label);
        crt_val_t **dev_sign_out = m_sign_gadget->get_dev_out_label();
        DevDevCopy2<<<nr_blocks, 32>>>(m_dev_out_label[0], dev_sign_out[0],
                                       m_input_size, nr_components);
        cudaDeviceSynchronize();

        // Step 4: Downshift by max_crt_modulus / 4
        crt_val_t **dev_downshift_labels = m_gc->get_dev_downshift_labels();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);

            SubLabel<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_out_label[i], m_dev_out_label[i], dev_downshift_labels[i],
                modulus, m_input_size, nr_comps);

            cudaStreamDestroy(stream[i]);
        }
        cudaDeviceSynchronize();

        return m_dev_out_label;
    }

    void cuda_move_output(vector<LabelTensor *> *out_label)
    {
        size_t crt_base_size = m_gc->get_crt_base().size();
        size_t nr_outputs = m_input_size;
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            cudaCheckError(cudaMemcpy(out_label->at(i)->get_components(),
                                      m_dev_out_label[i],
                                      nr_outputs * nr_comps * sizeof(crt_val_t),
                                      cudaMemcpyDeviceToHost));
        }
    }

#else

    void cuda_move()
    {
        // Allocate memory fo the output labels
        //// Allocate pointer array
        m_dev_out_label = new crt_val_t *[m_gc->get_crt_base().size()];
        int size = m_gc->get_crt_base().size() * sizeof(crt_val_t *);
        ocall_alloc_ptr_array(reinterpret_cast<void ***>(&m_dev_out_label),
                              size);
        //// Allocate device array
        size_t output_size = m_input_size;
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t size = nr_comps * output_size * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void **>(&m_dev_out_label[i]),
                             size);
        }

        // Move ciphertexts of proj_gates_trans_mod
        size = m_input_size * (crt_base_size - 1) * 2;
        ocall_cudaMalloc(reinterpret_cast<void **>(&m_dev_trans_mod_ciphers),
                         size * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_trans_mod_ciphers, m_trans_mod_ciphers,
                               size * sizeof(__uint128_t));

        // Move sign gadget
        m_sign_gadget->cuda_move();
    }

    crt_val_t **cuda_evaluate(crt_val_t **dev_in)
    {
        crt_val_t **dev_upshift_labels = m_gc->get_dev_upshift_labels();
        auto crt_base = m_gc->get_crt_base().data();
        size_t crt_base_size = m_gc->get_crt_base().size();
        crt_val_t *dev_zero_label = m_gc->get_dev_zero_label(2);
        ocall_eval_rescale(m_dev_out_label, dev_in, dev_upshift_labels,
                           dev_zero_label, m_dev_trans_mod_ciphers, crt_base,
                           crt_base_size, m_input_size);

        //// Sign gadget
        m_sign_gadget->cuda_evaluate(m_dev_out_label);
        crt_val_t **dev_sign_out = m_sign_gadget->get_dev_out_label();
        crt_val_t **dev_downshift_labels = m_gc->get_dev_downshift_labels();
        ocall_eval_rescale2(m_dev_out_label, dev_sign_out[0],
                            dev_downshift_labels, m_input_size, crt_base,
                            crt_base_size);

        return m_dev_out_label;
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output(vector<LabelTensor *> *out_label)
    {
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t *comps = out_label->at(i)->get_components();
            size_t size = out_label->at(i)->size();
            crt_val_t *tmp;
            ocall_cudaMemcpyFromDevice(reinterpret_cast<void **>(&tmp),
                                       m_dev_out_label[i],
                                       size * sizeof(crt_val_t));
            std::memcpy(comps, tmp, size * sizeof(crt_val_t));
            ocall_free(reinterpret_cast<void *>(tmp));
        }
    }

#endif

    crt_val_t **get_dev_out_label() { return m_dev_out_label; }

    void debug_print_all_constants() const
    {
        std::cerr << "RescaleGadget constants:" << std::endl;
        std::cerr << "m_input_size: " << m_input_size << std::endl;
        std::cerr << "m_scaling_factors: ";
        for (auto m : m_scaling_factors)
        {
            std::cerr << m << " ";
        }
        std::cerr << std::endl;
        std::cerr << "m_scaling_factors_indices: ";
        for (auto m : m_scaling_factors_indices)
        {
            std::cerr << m << " ";
        }
        std::cerr << std::endl;
    }
};

#endif // RESCALING_GADGET_H
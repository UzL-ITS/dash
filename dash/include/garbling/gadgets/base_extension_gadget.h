#ifndef BASE_EXTENSION_GADGET_H
#define BASE_EXTENSION_GADGET_H

#include <vector>
#include <numeric>
#include <cassert>
#include <algorithm>
#include <unordered_set>

#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/projection_gate.h"
#include "misc/datatypes.h"
#include "misc/misc.h"

using std::vector;

/**
 * @class BaseExtensionGadget
 * @brief This gadget is used to extend the base of a CRT representation by a set of moduli.
 * @author Felix Maurer
 */
class BaseExtensionGadget
{

public:
    BaseExtensionGadget(GarbledCircuitInterface *gc, const size_t nr_inputs, const vector<crt_val_t> out_moduli, const vector<crt_val_t> extra_moduli)
        : m_gc{gc},
          M_NR_INPUTS{nr_inputs},
          M_OUT_MODULI{out_moduli},
          M_EXTENDED_CRT_BASE_SIZE{out_moduli.size()},
          M_EXTRA_MODULI{extra_moduli},
          M_NON_EXTENDED_CRT_BASE_SIZE{out_moduli.size() - extra_moduli.size()},
          M_EXTRA_MODULI_INDICES{get_extra_moduli_indices(out_moduli, extra_moduli)}
    {
        // Assert that extra_moduli are a subset of out_moduli
        for (const auto modulus : M_EXTRA_MODULI)
        {
            assert(std::find(out_moduli.begin(), out_moduli.end(), modulus) != out_moduli.end());
        }

        // Assert no duplicates in extra_moduli
        std::unordered_set mod_set(M_EXTRA_MODULI.begin(), M_EXTRA_MODULI.end());
        assert(mod_set.size() == M_EXTRA_MODULI.size());

        // Can't be initialized in initializer list due to dependency on M_EXTRA_MODULI_INDICES
        m_out_moduli_swapped = get_out_moduli_swapped();
        m_inverse_mods_of_inverse_mods_total_base = get_inverse_mods_of_inverse_mods_total_base();
        m_inverse_mods_partial_base = get_inverse_mods_partial_bases();
    }

    ~BaseExtensionGadget()
    {
        for (auto proj_gates_loop : m_proj_gates_base_change_loop)
        {
            for (auto proj_gates : proj_gates_loop)
            {
                for (auto proj_gate : proj_gates)
                {
                    delete proj_gate;
                }
            }
        }

        if (m_proj_gates_base_change_loop_ciphers != nullptr)
        {
            delete[] m_proj_gates_base_change_loop_ciphers;
        }
    }

    void garble(vector<LabelTensor *> *in_label,
                vector<LabelTensor *> *out_label)
    {
        // Assert dimension of input and output labels
        assert(in_label->size() == M_EXTENDED_CRT_BASE_SIZE);
        assert(out_label->size() == M_EXTENDED_CRT_BASE_SIZE);

        // Compute total cipher count dynamically:
        size_t total_cipher_count = 0;
        for (size_t i = 0; i < M_NON_EXTENDED_CRT_BASE_SIZE; ++i)
        {
            // For each i, we perform (M_EXTENDED_CRT_BASE_SIZE - i - 1) projections,
            // and each projection uses m_out_moduli_swapped[i] number of __uint128_t's.
            total_cipher_count += (M_EXTENDED_CRT_BASE_SIZE - i - 1) * m_out_moduli_swapped[i];
        }
        total_cipher_count *= M_NR_INPUTS;

        // 0. Allocate ciphers and resize gate vectors
        m_proj_gates_base_change_loop_ciphers = new __uint128_t[total_cipher_count];
        size_t bc_loop_cipher_cnt = 0;
        m_proj_gates_base_change_loop.resize(M_NON_EXTENDED_CRT_BASE_SIZE);

        // 1. Init variables
        // 1.1 in_label_swapped
        vector<LabelTensor *> in_label_swapped = *in_label;
        for (size_t i = 1; i <= M_EXTRA_MODULI_INDICES.size(); ++i)
        {
            // select ith last extra modulus index
            const auto extra_mod_idx = M_EXTRA_MODULI_INDICES[M_EXTRA_MODULI_INDICES.size() - i];
            // swap it to the ith last position of the input
            std::swap(in_label_swapped[extra_mod_idx], in_label_swapped[in_label_swapped.size() - i]);
        }

        // 1.2 Copy in_label_swapped to l_w
        vector<LabelTensor> l_w{};
        l_w.resize(M_EXTENDED_CRT_BASE_SIZE);
        for (size_t i = 0; i < M_EXTENDED_CRT_BASE_SIZE; ++i)
        {
            l_w.at(i) = LabelTensor{m_out_moduli_swapped.at(i), in_label->at(0)->get_dims()};

#ifndef SGX
#pragma omp parallel for
#endif
            for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
            {
                l_w.at(i).set_label(in_label_swapped.at(i)->get_label(input_idx), input_idx);
            }
        }

        // 2. MRS algorithm
        for (size_t i = 0; i < M_NON_EXTENDED_CRT_BASE_SIZE; ++i)
        {
            // 2.1 Eliminate ith modulus
            m_proj_gates_base_change_loop[i].resize(M_EXTENDED_CRT_BASE_SIZE - i - 1);
            for (size_t j = 0; j < M_EXTENDED_CRT_BASE_SIZE - i - 1; ++j)
            {
                m_proj_gates_base_change_loop[i][j].resize(M_NR_INPUTS);
                for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
                {
                    // BC_LOOP (l_w[j+i+1] -= l_w[i])
                    auto proj_out_base_label_loop = new LabelTensor{m_out_moduli_swapped[j + i + 1]};
                    proj_out_base_label_loop->init_random();

                    ProjectionGate *proj_gate_bc_loop = new ProjectionGate(
                        m_out_moduli_swapped[j + i + 1],
                        &m_proj_gates_base_change_loop_ciphers[bc_loop_cipher_cnt],
                        &ProjectionFunctionalities::identity,
                        m_gc);
                    proj_gate_bc_loop->garble(
                        l_w.at(i).get_label(input_idx),
                        proj_out_base_label_loop,
                        m_gc->get_label_offset(m_out_moduli_swapped[j + i + 1]),
                        nullptr);

                    LabelSlice subtraction_slice = l_w.at(j + i + 1).get_label(input_idx);
                    subtraction_slice -= proj_out_base_label_loop->get_label(0);

                    m_proj_gates_base_change_loop[i][j][input_idx] = proj_gate_bc_loop;
                    bc_loop_cipher_cnt += l_w.at(i).get_modulus();

                    subtraction_slice *= m_inverse_mods_partial_base[i][j];
                }
            }
        }

        // 3. Write to out_label
        for (size_t i = 0; i < M_EXTENDED_CRT_BASE_SIZE; ++i)
        {

            if (std::find(M_EXTRA_MODULI_INDICES.begin(), M_EXTRA_MODULI_INDICES.end(), i) != M_EXTRA_MODULI_INDICES.end())
            {
#ifndef SGX
#pragma omp parallel for
#endif
                for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
                {
                    // find position of ith modulus in m_out_moduli_swapped
                    const auto iter = std::find(m_out_moduli_swapped.begin(), m_out_moduli_swapped.end(), M_OUT_MODULI.at(i));
                    const auto i_swapped = std::distance(m_out_moduli_swapped.begin(), iter);

                    out_label->at(i)->set_label(
                        l_w.at(i_swapped).get_label(input_idx),
                        input_idx);
                }
            }
            else
            {
#ifndef SGX
#pragma omp parallel for
#endif
                for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
                {
                    out_label->at(i)->set_label(
                        in_label->at(i)->get_label(input_idx),
                        input_idx);
                }
            }
        }

        // 4.+ 5. Solve linear equation on extra moduli
        for (size_t i = 0; i < M_EXTRA_MODULI.size(); ++i)
        {
            const auto crt_idx = M_EXTRA_MODULI_INDICES[i];
            *out_label->at(crt_idx) *= -m_inverse_mods_of_inverse_mods_total_base[i];
        }
    }

    void cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                      vector<LabelTensor *> *out_label,
                      int nr_threads)
    {
        // Assert dimension of input and output labels
        assert(encoded_inputs->size() == M_EXTENDED_CRT_BASE_SIZE);
        assert(out_label->size() == M_EXTENDED_CRT_BASE_SIZE);

        // 1. Init variables
        // 1.1 in_label_swapped
        vector<LabelTensor *> encoded_inputs_swapped = *encoded_inputs;
        for (size_t i = 1; i <= M_EXTRA_MODULI_INDICES.size(); ++i)
        {
            // select ith last extra modulus index
            const auto extra_mod_idx = M_EXTRA_MODULI_INDICES[M_EXTRA_MODULI_INDICES.size() - i];
            // swap it to the ith last position of the input
            std::swap(encoded_inputs_swapped[extra_mod_idx], encoded_inputs_swapped[encoded_inputs_swapped.size() - i]);
        }

        // 1.2 Copy encoded_inputs_swapped to l_w
        vector<LabelTensor> l_w{};
        l_w.resize(M_EXTENDED_CRT_BASE_SIZE);
        for (size_t i = 0; i < M_EXTENDED_CRT_BASE_SIZE; ++i)
        {
            l_w.at(i) = LabelTensor{m_out_moduli_swapped.at(i), encoded_inputs->at(0)->get_dims()};

#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
            for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
            {
                l_w.at(i).set_label(encoded_inputs_swapped.at(i)->get_label(input_idx), input_idx);
            }
        }

        // 2. MRS algorithm
        for (size_t i = 0; i < M_NON_EXTENDED_CRT_BASE_SIZE; ++i)
        {
            // 2.1 Eliminate ith modulus
            for (size_t j = 0; j < M_EXTENDED_CRT_BASE_SIZE - i - 1; ++j)
            {
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
                for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
                {
                    // BC_LOOP (l_w[j+i+1] -= l_w[i])
                    ProjectionGate *proj_gate_bc_loop = m_proj_gates_base_change_loop[i][j][input_idx];
                    LabelTensor proj_bc_loop_out{m_out_moduli_swapped[j + i + 1]};
                    proj_bc_loop_out = proj_gate_bc_loop->cpu_evaluate(l_w.at(i).get_label(input_idx));

                    LabelSlice subtraction_slice = l_w.at(j + i + 1).get_label(input_idx);
                    subtraction_slice -= proj_bc_loop_out.get_label(0);

                    subtraction_slice *= m_inverse_mods_partial_base[i][j];
                }
            }
        }

        // 3. Write to out_label
        for (size_t i = 0; i < M_EXTENDED_CRT_BASE_SIZE; ++i)
        {
            if (std::find(M_EXTRA_MODULI_INDICES.begin(), M_EXTRA_MODULI_INDICES.end(), i) != M_EXTRA_MODULI_INDICES.end())
            {
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
                for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
                {
                    // find position of ith modulus in m_out_moduli_swapped
                    const auto iter = std::find(m_out_moduli_swapped.begin(), m_out_moduli_swapped.end(), M_OUT_MODULI.at(i));
                    const auto i_swapped = std::distance(m_out_moduli_swapped.begin(), iter);

                    out_label->at(i)->set_label(
                        l_w.at(i_swapped).get_label(input_idx),
                        input_idx);
                }
            }
            else
            {
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
                for (size_t input_idx = 0; input_idx < M_NR_INPUTS; ++input_idx)
                {
                    out_label->at(i)->set_label(
                        encoded_inputs->at(i)->get_label(input_idx),
                        input_idx);
                }
            }
        }

        // 4.+ 5. Solve linear equation on extra moduli
        for (size_t i = 0; i < M_EXTRA_MODULI.size(); ++i)
        {
            const auto crt_idx = M_EXTRA_MODULI_INDICES[i];
            *out_label->at(crt_idx) *= -m_inverse_mods_of_inverse_mods_total_base[i];
        }
    }

    void cuda_move()
    {
        throw "Not yet implemented!";
    }

    void cuda_evaluate(crt_val_t **dev_in_label)
    {
        throw "Not yet implemented!";
    }

    void cuda_move_output()
    {
        throw "Not yet implemented!";
    }

    crt_val_t **get_dev_out_label()
    {
        throw "Not yet implemented!";
    }

    vector<crt_val_t> get_out_moduli() const
    {
        return m_out_moduli_swapped;
    }

private:
    inline vector<size_t> get_extra_moduli_indices(const vector<crt_val_t> out_moduli, const vector<crt_val_t> extra_moduli) const
    {
        vector<size_t> indices;
        for (const auto modulus : extra_moduli)
        {
            indices.push_back((size_t)(std::find(out_moduli.begin(), out_moduli.end(), modulus) - out_moduli.begin()));
        }
        return indices;
    }

    //TODO: this doesnt work if M_EXTRA_MODULI.size() > M_OUT_MODULI.size() / 2 => find better solution (also adjust swapping of inputs in garble+eval)
    inline vector<crt_val_t> get_out_moduli_swapped() const
    {
        vector<crt_val_t> out_moduli_swapped = M_OUT_MODULI;
        for (size_t i = 1; i <= M_EXTRA_MODULI_INDICES.size(); ++i)
        {
            // select ith last extra modulus index
            const auto extra_mod_idx = M_EXTRA_MODULI_INDICES[M_EXTRA_MODULI_INDICES.size() - i];
            // swap it to the ith last position of m_out_moduli_swapped
            std::swap(out_moduli_swapped[extra_mod_idx], out_moduli_swapped[out_moduli_swapped.size() - i]);
        }
        return out_moduli_swapped;
    }

    inline vector<crt_val_t> get_inverse_mods_of_inverse_mods_total_base() const
    {
        vector<crt_val_t> invvs = {};
        for (const auto modulus : M_EXTRA_MODULI)
        {
            // Note that acc explodes very fast for larger bases. unsigned long supports up to the product of the first 9 primes
            const unsigned long acc = std::accumulate(m_out_moduli_swapped.begin(), m_out_moduli_swapped.end() - M_EXTRA_MODULI.size(), 1, std::multiplies<unsigned long>());
            const crt_val_t inverse_mod_total_base = util::mul_inv(acc, modulus);
            const auto invv = util::mul_inv(inverse_mod_total_base, modulus);
            invvs.push_back(invv);
        }

        return invvs;
    }

    inline vector<vector<crt_val_t>> get_inverse_mods_partial_bases() const
    {
        vector<vector<crt_val_t>> inv_mods_partial_bases;
        for (size_t i = 0; i < M_EXTENDED_CRT_BASE_SIZE - 1; ++i)
        {
            vector<crt_val_t> inv_b_i;
            for (size_t j = i + 1; j < M_EXTENDED_CRT_BASE_SIZE; ++j)
            {
                const crt_val_t inv = util::mul_inv(m_out_moduli_swapped[i], m_out_moduli_swapped[j]);
                inv_b_i.push_back(inv);
            }
            inv_mods_partial_bases.push_back(inv_b_i);
        }

        return inv_mods_partial_bases;
    }

    GarbledCircuitInterface *m_gc;
    vector<crt_val_t> m_out_moduli_swapped;
    vector<crt_val_t> m_inverse_mods_of_inverse_mods_total_base;
    vector<vector<crt_val_t>> m_inverse_mods_partial_base;

    // Projection Gates (initliazed in garble() method)
    // size: M_NON_EXTENDED_CRT_BASE_SIZE * M_NR_INPUTS
    vector<vector<vector<ProjectionGate *>>> m_proj_gates_base_change_loop;
    // size: M_NON_EXTENDED_CRT_BASE_SIZE * M_NR_INPUTS * M_EXTRA_MODULUS
    __uint128_t *m_proj_gates_base_change_loop_ciphers;

    const size_t M_NR_INPUTS;
    const vector<crt_val_t> M_OUT_MODULI;
    const size_t M_EXTENDED_CRT_BASE_SIZE;
    const size_t M_NON_EXTENDED_CRT_BASE_SIZE;
    const vector<crt_val_t> M_EXTRA_MODULI;
    const vector<size_t> M_EXTRA_MODULI_INDICES;
};

#endif // BASE_EXTENSION_GADGET_H
#ifndef RELU_GADGET_H
#define RELU_GADGET_H

#include <cstdlib>
#include <cstring>
#include <vector>

#include "garbling/gates/mixed_mod_half_gate.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gadgets/sign_gadget.h"
#include "misc/datatypes.h"
#include "misc/misc.h"

using std::vector;

/**
 * @class ReluGadget
 * @brief The ReluGadget class represents a gadget for performing the Rectified Linear Unit (ReLU) operation.
 * @author Felix Maurer
 *
 * The ReluGadget class provides methods for garbling and evaluating ReLU circuits, as well as CUDA acceleration.
 * It uses the SignGadget class to compute the sign of the inputs and performs multiplication using MixedModHalfGate objects.
 *
 * The garble() method garbles the ReLU circuit given input labels and produces output labels.
 * The cpu_evaluate() method evaluates the ReLU circuit on the CPU given encoded inputs and produces output labels.
 * The cuda_move(), cuda_evaluate(), and cuda_move_output() methods are placeholders for CUDA acceleration and are not yet implemented.
 */
class ReluGadget
{
public:
    ReluGadget(GarbledCircuitInterface *gc, const size_t nr_inputs)
        : m_gc{gc},
          M_NR_INPUTS{nr_inputs},
          M_CRT_BASE{m_gc->get_crt_base()},
          M_CRT_BASE_SUM{std::accumulate(M_CRT_BASE.begin(), M_CRT_BASE.end(), 0)},
          M_GARBLER_CIPHERS{new __uint128_t[M_NR_INPUTS * M_CRT_BASE_SUM]},
          M_EVALUATOR_CIPHERS{new __uint128_t[M_NR_INPUTS * M_CRT_BASE.size() * (/*q=*/2 + 1)]},
          m_sign_gadget{new SignGadget{m_gc, 0, 1, M_NR_INPUTS, vector<crt_val_t> {2}}}
    {
    }

    ~ReluGadget()
    {
        delete m_sign_gadget;
        for (auto& mm_hgs : m_mixed_mod_half_gates)
        {
            for (auto mmhg : mm_hgs)
            {
                delete mmhg;
            }
        }
        delete[] M_GARBLER_CIPHERS;
        delete[] M_EVALUATOR_CIPHERS;
    }

    void garble(vector<LabelTensor *> *in_label,
                vector<LabelTensor *> *out_label)
    {
        const dim_t input_dims = in_label->at(0)->get_dims();

        // Relu(x) := x * Sign(x)
        // 1. Sign
        // 1.1 Init out label
        auto label = new LabelTensor{2, input_dims};
        label->init_random();
        vector<LabelTensor *> out_label_sign{label};
        // 1.2 Apply Sign gadget
        m_sign_gadget->garble(in_label, &out_label_sign);

        // 2. Multiplication
        m_mixed_mod_half_gates.resize(M_NR_INPUTS);
        for (size_t i = 0; i < M_NR_INPUTS; ++i)
        {
            int crt_base_prefix = 0;

            vector<MixedModHalfGate *> mm_hgs;
            for (size_t j = 0; j < M_CRT_BASE.size(); ++j)
            {
                const crt_val_t q = 2;
                const crt_val_t p = M_CRT_BASE.at(j);

                // Get cipher indices
                auto *g_c = &M_GARBLER_CIPHERS[i * M_CRT_BASE_SUM + crt_base_prefix];
                auto *e_c = &M_EVALUATOR_CIPHERS[i * M_CRT_BASE.size() * (q + 1) + j * (q + 1)];

                auto mmhg = new MixedModHalfGate{q, p, g_c, e_c, m_gc};
                mm_hgs.push_back(mmhg);

                auto result = mmhg->garble(in_label->at(j)->get_label(i),
                                           out_label_sign.at(0)->get_label(i));
                out_label->at(j)->set_label(result.get_label(0), i);

                crt_base_prefix += p;
            }
            m_mixed_mod_half_gates.at(i) = mm_hgs;
        }

        for (auto label : out_label_sign)
        {
            delete label;
        }
    }

    void cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                                        vector<LabelTensor *> *out_label, int nr_threads)
    {
        const dim_t input_dims = encoded_inputs->at(0)->get_dims();

        // Relu(x) := x * Sign(x)
        // 1. Sign
        // 1.1 Init out label
        auto label = new LabelTensor{2, input_dims}; // input dims == output dims
        label->init_random();
        vector<LabelTensor *> out_label_sign{label};
        // 1.2 Apply Sign gadget
        m_sign_gadget->cpu_evaluate(encoded_inputs, &out_label_sign, nr_threads);

        // 2. Multiplication
        m_mixed_mod_half_gates.resize(M_NR_INPUTS);
        for (size_t i = 0; i < M_NR_INPUTS; ++i)
        {
            vector<MixedModHalfGate *> mm_hgs;
            for (size_t j = 0; j < M_CRT_BASE.size(); ++j)
            {
                auto mmhg = m_mixed_mod_half_gates.at(i).at(j);

                auto result = mmhg->cpu_evaluate(encoded_inputs->at(j)->get_label(i),
                                                 out_label_sign.at(0)->get_label(i));
                out_label->at(j)->set_label(result.get_label(0), i);
            }
        }

        for (auto label : out_label_sign)
        {
            delete label;
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

private:
    GarbledCircuitInterface *m_gc;

    const size_t M_NR_INPUTS;
    const vector<crt_val_t> M_CRT_BASE;
    const int M_CRT_BASE_SUM;

    static const int M_SIGN_LOWER_BOUND = 0;
    static const int M_SIGN_UPPER_BOUND = 1;

    SignGadget *m_sign_gadget;

    // size: nr_inputs x crt_base_size
    vector<vector<MixedModHalfGate *>> m_mixed_mod_half_gates;
    // size: nr_inputs * crt_base_sum
    __uint128_t *M_GARBLER_CIPHERS;
    // size: crt_base_size * nr_inputs * (q + 1), with q = 2
    __uint128_t *M_EVALUATOR_CIPHERS;
};

#endif // RELU_GADGET_H
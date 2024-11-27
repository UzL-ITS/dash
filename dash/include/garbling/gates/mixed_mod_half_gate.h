#ifndef MIXED_MOD_HG_H
#define MIXED_MOD_HG_H

#include "misc/datatypes.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/generalized_half_gate.h"
#include "garbling/label_tensor.h"
#include "misc/misc.h"
#include "garbling/gates/projection_gate.h"
#include "garbling/gates/projection_gate_mini.h"

typedef struct mini_evaluator_hg_params {
    crt_val_t payload_modulus;
    crt_val_t r;
} mini_evaluator_hg_params_t;

crt_val_t mini_evaluator_hg_functionality(crt_val_t y, void* params) {
    mini_evaluator_hg_params_t* p = (mini_evaluator_hg_params_t*)params;
    return util::modulo<crt_val_t>((y + p->r), p->payload_modulus);
}

class MixedModHalfGate {
    ProjectionGate* m_garbler_hg{nullptr};
    __uint128_t* m_garbler_ciphers;
    ProjectionGate* m_evaluator_hg{nullptr};
    ProjectionGateMini* m_mini_evaluator_hg{nullptr};
    __uint128_t* m_evaluator_ciphers;

    crt_val_t m_modulus_p;  // modulus of the garbler half gate
    crt_val_t m_modulus_q;  // modulus of the evaluator half gate
    GarbledCircuitInterface* m_gc;

   public:
    MixedModHalfGate(crt_val_t modulus_q, crt_val_t modulus_p,
                     __uint128_t* garbler_ciphers,
                     __uint128_t* evaluator_ciphers,
                     GarbledCircuitInterface* gc)
        : m_modulus_q{modulus_q},
          m_modulus_p{modulus_p},
          m_garbler_ciphers{garbler_ciphers},
          m_evaluator_ciphers{evaluator_ciphers},
          m_gc{gc} {}

    ~MixedModHalfGate() {
        if (m_garbler_hg != nullptr) {
            delete m_garbler_hg;
        }
        if (m_evaluator_hg != nullptr) {
            delete m_evaluator_hg;
        }
        if (m_mini_evaluator_hg != nullptr) {
            delete m_mini_evaluator_hg;
        }
    }

    LabelTensor garble(LabelSlice sk01, LabelSlice sk02) {
        // Initialize projection gates
        m_garbler_hg = new ProjectionGate(m_modulus_p, m_garbler_ciphers,
                                          &garbler_hg_functionality, m_gc);

        m_evaluator_hg = new ProjectionGate(m_modulus_p, m_evaluator_ciphers,
                                            &evaluator_hg_functionality, m_gc);

        m_mini_evaluator_hg = new ProjectionGateMini(
            m_modulus_q, &m_evaluator_ciphers[m_modulus_q],
            &mini_evaluator_hg_functionality, m_gc);

        // Garble garbler half-gate
        crt_val_t r = sk01.get_color();
        LabelTensor sk03{m_modulus_p};
        sk03.init_random();
        m_garbler_hg->garble(sk01, &sk03, m_gc->get_label_offset(m_modulus_p),
                             (void*)&r);

        // Garble evaluator half-gates
        evaluator_hg_params_t params{.out_modulus = m_modulus_p, .r = r};
        LabelTensor sk04{m_modulus_p};
        sk04.init_random();
        m_evaluator_hg->garble(sk02, &sk04, sk01, (void*)&params);

        // - Mini projection gate for r+y
        mini_evaluator_hg_params_t mini_params{.payload_modulus = m_modulus_p,
                                               .r = r};
        m_mini_evaluator_hg->garble(sk02, (void*)&mini_params);

        // Compute output
        sk04 -= sk03;
        return sk04;
    }

    LabelTensor cpu_evaluate(LabelSlice sk01_xR, LabelSlice sk02_yR) {
        // Decrypt garbler half-gate
        auto garbler_label = m_garbler_hg->cpu_evaluate(sk01_xR);

        // Decrypt evaluator half-gate
        auto evaluator_label = m_evaluator_hg->cpu_evaluate(sk02_yR);

        // - Decrypt mini projection gate for r+y
        crt_val_t y_plus_r = m_mini_evaluator_hg->cpu_evaluate(sk02_yR);
        auto result = LabelTensor{evaluator_label};
        result += (y_plus_r * sk01_xR);
        result -= garbler_label;
        return result;
    }
};

#endif
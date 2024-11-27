#ifndef GENERALIZED_HALF_GATE_H
#define GENERALIZED_HALF_GATE_H

#include "misc/datatypes.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/label_tensor.h"
#include "misc/misc.h"
#include "garbling/gates/projection_gate.h"

crt_val_t garbler_hg_functionality(crt_val_t x, void* params) {
    crt_val_t r = *(crt_val_t*)params;
    return x * r;
}

typedef struct evaluator_hg_params {
    crt_val_t out_modulus;
    crt_val_t r;
} evaluator_hg_params_t;

crt_val_t evaluator_hg_functionality(crt_val_t y, void* params) {
    evaluator_hg_params_t* p = (evaluator_hg_params_t*)params;
    return util::modulo<crt_val_t>(-(y + p->r), p->out_modulus);
}

class GeneralizedHalfGate {
    ProjectionGate* m_garbler_hg{nullptr};
    __uint128_t* m_garbler_ciphers;
    ProjectionGate* m_evaluator_hg{nullptr};
    __uint128_t* m_evaluator_ciphers;
    crt_val_t m_modulus;
    GarbledCircuitInterface* m_gc;

   public:
    GeneralizedHalfGate(crt_val_t modulus, __uint128_t* garbler_ciphers,
                        __uint128_t* evaluator_ciphers,
                        GarbledCircuitInterface* gc)
        : m_modulus{modulus},
          m_garbler_ciphers{garbler_ciphers},
          m_evaluator_ciphers{evaluator_ciphers},
          m_gc{gc} {}

    ~GeneralizedHalfGate() {
        if (m_garbler_hg != nullptr) {
            delete m_garbler_hg;
        }
        if (m_evaluator_hg != nullptr) {
            delete m_evaluator_hg;
        }
    }

    LabelTensor garble(LabelSlice sk01, LabelSlice sk02) {
        // Initialize projection gates - one per half gate
        m_garbler_hg = new ProjectionGate(m_modulus, m_garbler_ciphers,
                                          &garbler_hg_functionality, m_gc);

        m_evaluator_hg = new ProjectionGate(m_modulus, m_evaluator_ciphers,
                                            &evaluator_hg_functionality, m_gc);

        // Garble garbler half-gate
        crt_val_t r = sk02.get_color();
        LabelTensor sk03{m_modulus};
        sk03.init_random();
        m_garbler_hg->garble(sk01, &sk03, m_gc->get_label_offset(m_modulus),
                             (void*)&r);

        // Garble evaluator half-gate
        evaluator_hg_params_t params{.out_modulus = m_modulus, .r = r};
        LabelTensor sk04{m_modulus};
        sk04.init_random();
        m_evaluator_hg->garble(sk02, &sk04, sk01, (void*)&params);

        // Compute output
        sk04 -= sk03;
        return sk04;
    }

    LabelTensor cpu_evaluate(LabelSlice sk01_xR, LabelSlice sk02_yR) {
        // Decrypt garbler half-gate
        auto garbler_label = m_garbler_hg->cpu_evaluate(sk01_xR);

        // Decrypt evaluator half-gate
        auto evaluator_label = m_evaluator_hg->cpu_evaluate(sk02_yR);

        crt_val_t y_plus_r = sk02_yR.get_color();
        auto result = LabelTensor{evaluator_label};
        result += (y_plus_r * sk01_xR);
        result -= garbler_label;
        return result;
    }
};

#endif
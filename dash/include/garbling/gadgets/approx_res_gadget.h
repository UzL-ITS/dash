#ifndef APPROX_RES_GADGET_H
#define APPROX_RES_GADGET_H

#include <vector>

#include "garbling/gadgets/lookup_approx_sign.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/projection_gate.h"
#include "misc/datatypes.h"

#ifndef SGX
#include <cuda_runtime_api.h>
#endif

using std::vector;

typedef struct approx_lookup_params {
    LookupApproxSign* lookup;
    int crt_base_idx;
    int mrs_base_idx;
} approx_lookup_params_t;

crt_val_t approx_lookup_functionality(crt_val_t residue, void* params) {
    auto p = static_cast<approx_lookup_params*>(params);
    auto value = p->lookup->get(p->crt_base_idx, residue)[p->mrs_base_idx];
    return value;
}

/**
 * @brief Approximated lookup needed in the first step of approx. sign gadget.
 *
 * Approximate resiude and represent in mixed radix representation.
 * 
 */
class ApproxResGadget {
    GarbledCircuitInterface* m_gc;
    crt_val_t m_in_modulus;
    vector<ProjectionGate*> m_proj_gates_approx;
    __uint128_t* m_approx_ciphers;

#ifndef SGX
    crt_val_t* m_dev_out_label{nullptr};
    __uint128_t* m_dev_approx_ciphers{nullptr};
#endif

   public:
    ApproxResGadget(GarbledCircuitInterface* gc, __uint128_t* ciphers,
                    crt_val_t in_modulus)
        : m_gc{gc}, m_approx_ciphers{ciphers}, m_in_modulus{in_modulus} {}

    ~ApproxResGadget() {
        for (auto& p : m_proj_gates_approx) {
            delete p;
        }
#ifndef SGX
        if (m_dev_approx_ciphers != nullptr) {
            free(m_dev_approx_ciphers);
        }
        if (m_dev_out_label != nullptr) {
            free(m_dev_out_label);
        }
#endif
    }

    vector<LabelTensor*> garble(LabelSlice in_label, int crt_base_idx) {
        int mrs_base_size = m_gc->get_mrs_base().size();

        vector<LabelTensor*> out_labels;
        out_labels.reserve(mrs_base_size);

        m_proj_gates_approx.reserve(mrs_base_size);
        approx_lookup_params_t tmp;
        tmp.lookup = m_gc->get_lookup_approx_sign();
        for (int i = 0; i < mrs_base_size; ++i) {
            tmp.crt_base_idx = crt_base_idx;
            tmp.mrs_base_idx = i;
            crt_val_t out_modulus = m_gc->get_mrs_base()[i];
            auto proj_gate = new ProjectionGate(
                out_modulus, m_approx_ciphers + i * m_in_modulus,
                &approx_lookup_functionality, m_gc);

            auto out_base_label = new LabelTensor{out_modulus};
            out_base_label->init_random();
            auto out_offset_label = m_gc->get_label_offset(out_modulus);
            proj_gate->garble(in_label, out_base_label, out_offset_label, &tmp);
            out_labels.push_back(out_base_label);

            m_proj_gates_approx.push_back(proj_gate);
        }

        return out_labels;
    }

    vector<LabelTensor*> cpu_evaluate(LabelSlice g_input) {
        int mrs_base_size = m_gc->get_mrs_base().size();
        vector<LabelTensor*> out_labels;
        for (int i = 0; i < mrs_base_size; ++i) {
            auto out_label = m_proj_gates_approx[i]->cpu_evaluate(g_input);
            out_labels.push_back(new LabelTensor{out_label});
        }
        return out_labels;
    }
};

#endif
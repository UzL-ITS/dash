#ifndef PROJECTION_GATE_MINI_H
#define PROJECTION_GATE_MINI_H

#include "garbling/garbled_circuit_interface.h"
#include "garbling/label_tensor.h"
#include "misc/datatypes.h"
#include "misc/misc.h"

class ProjectionGateMini {
    __uint128_t* m_cipher;
    crt_val_t m_out_modulus;
    crt_val_t (*m_functionality)(crt_val_t, void*);
    void* m_functionality_args;
    GarbledCircuitInterface* m_gc;

   public:
    ProjectionGateMini(crt_val_t out_modulus, __uint128_t* cipher,
                       crt_val_t (*functionality)(crt_val_t, void*),
                       GarbledCircuitInterface* gc)
        : m_out_modulus{out_modulus},
          m_cipher{cipher},
          m_functionality{functionality},
          m_gc{gc} {}

    void garble(LabelSlice in_label, void* functionality_args = nullptr) {
        crt_val_t in_modulus{in_label.get_modulus()};

        for (int i = 0; i < in_modulus; ++i) {
            // Generate encryption key
            LabelTensor key{in_label};
            key += (i * m_gc->get_label_offset(in_modulus));
            key.compress();
            key.hash();

            // Generate payload
            crt_val_t x = m_functionality(i, functionality_args);

            // Encrypt payload
            int color_value = key.get_color(0);
            ((crt_val_t*)m_cipher)[color_value] =
                x + (crt_val_t)(key.get_hashed()[0]);
        }
    }

    crt_val_t cpu_evaluate(LabelSlice g_input) {
        // generate key
        LabelTensor key{g_input};
        key.compress();
        key.hash();

        // decrypt payload
        int color_value = key.get_color(0);
        crt_val_t payload = ((crt_val_t*)m_cipher)[color_value] -
                            (crt_val_t)(key.get_hashed()[0]);
        return payload;
    }
};

#endif
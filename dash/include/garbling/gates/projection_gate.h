#ifndef PROJECTION_GATE_H
#define PROJECTION_GATE_H

#include "garbling/garbled_circuit_interface.h"
#include "garbling/label_tensor.h"
#include "misc/datatypes.h"
#include "misc/misc.h"

// Enable the garbled-row-reduction optimization circuit-wide for alle
// projection gates.
// #define GRR3

namespace ProjectionFunctionalities {
crt_val_t identity(crt_val_t x, void* params) { return x; }
}  // namespace ProjectionFunctionalities

class ProjectionGate {
    __uint128_t* m_ciphers;
    crt_val_t m_out_modulus;
    crt_val_t (*m_functionality)(crt_val_t, void*);
    GarbledCircuitInterface* m_gc;

   public:
    ProjectionGate(crt_val_t out_modulus, __uint128_t* ciphers,
                   crt_val_t (*functionality)(crt_val_t, void*),
                   GarbledCircuitInterface* gc)
        : m_out_modulus{out_modulus},
          m_ciphers{ciphers},
          m_functionality{functionality},
          m_gc{gc} {}

    __uint128_t garble(LabelSlice in_label, void* functionality_args = nullptr) {
        crt_val_t in_modulus{in_label.get_modulus()};

        LabelTensor base{m_out_modulus};
        base.init_random();
        base.compress();
        __uint128_t ret = base.get_compressed()[0];
        for (int i = 0; i < in_modulus; ++i) {
            // Generate encryption key
            LabelTensor key{m_gc->get_label_offset(in_modulus)};
            key *= i;
            key += in_label;
            key.compress();
            key.hash();

            // Generate payload
            crt_val_t x = m_functionality(i, functionality_args);
            LabelTensor payload{m_gc->get_label_offset(m_out_modulus)};
            payload *= x;
            payload += base;
            payload.compress();

            // Encrypt payload
            int color_value = key.get_color(0);
            m_ciphers[color_value] =
                payload.get_compressed()[0] + key.get_hashed()[0];
        }
        return ret;
    }

    void garble(LabelSlice in_label, LabelTensor* out_base_label,
                const LabelSlice out_offset_label,
                void* functionality_args = nullptr) {
        crt_val_t in_modulus{in_label.get_modulus()};

        for (int i = 0; i < in_modulus; ++i) {
            // Generate encryption key
            LabelTensor key{m_gc->get_label_offset(in_modulus)};
            key *= i;
            key += in_label;
            key.compress();
            key.hash();
            crt_val_t color_value = key.get_color(0);

            // Generate payload
            crt_val_t x = m_functionality(i, functionality_args);
            LabelTensor payload{out_offset_label};
            payload *= x;
            payload += *out_base_label;
            payload.compress();

            // Encrypt payload
            m_ciphers[color_value] =
                payload.get_compressed()[0] + key.get_hashed()[0];
        }
    }

    LabelTensor cpu_evaluate(LabelSlice g_input) {
        // generate key
        LabelTensor key{g_input};
        key.compress();
        key.hash();

        // decrypt payload
        int color_value = key.get_color(0);
        __uint128_t payload = m_ciphers[color_value] - key.get_hashed()[0];
        LabelTensor g_output{m_out_modulus};
        g_output.set_compressed(payload, 0);
        g_output.decompress();
        return g_output;
    }
};

#endif
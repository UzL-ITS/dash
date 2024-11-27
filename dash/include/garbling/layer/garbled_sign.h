#ifndef GARBLED_SIGN_H
#define GARBLED_SIGN_H

#include <cstdlib>
#include <vector>

#include "garbling/gadgets/sign_gadget.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/layer/garbled_layer.h"
#include "garbling/layer/garbled_sign.h"
#include "misc/datatypes.h"
#include "misc/util.h"

using std::vector;

class GarbledSign : public GarbledLayer {
    SignGadget* m_sign_gadget;

   public:
    GarbledSign(Layer* layer_ptr, GarbledCircuitInterface* gc)
        : GarbledLayer{layer_ptr, gc} {
        m_sign_gadget = new SignGadget{m_gc, -1, 1, m_layer->get_input_size(),
                                       m_gc->get_crt_base()};
    }

    virtual ~GarbledSign() { delete m_sign_gadget; }

    void garble(vector<LabelTensor*>* in_label) override {
        // Reserve ouput_label
        size_t crt_base_size = m_gc->get_crt_base().size();
        dim_t output_dims = in_label->at(0)->get_dims();
        for (size_t i = 0; i < crt_base_size; ++i) {
            auto modulus = m_gc->get_crt_base().at(i);
            auto output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        m_sign_gadget->garble(in_label, m_out_label);
    }

    vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                                       int nr_threads) override {
        // Reserve ouput_label
        free_out_label();
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            dim_t output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        m_sign_gadget->cpu_evaluate(encoded_inputs, m_out_label, nr_threads);
        return m_out_label;
    }

    void cuda_move() override { m_sign_gadget->cuda_move(); }

    void cuda_evaluate(crt_val_t** dev_in_label) override {
        free_out_label();
        m_sign_gadget->cuda_evaluate(dev_in_label);
    }

    void cuda_move_output() override {
        // Reserve ouput_label
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            dim_t output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        m_sign_gadget->cuda_move_output(m_out_label);
    }

    crt_val_t** get_dev_out_label() override {
        return m_sign_gadget->get_dev_out_label();
    }
};

#endif
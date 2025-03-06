#ifndef GARBLED_BASE_EXTENSION_H
#define GARBLED_BASE_EXTENSION_H

#include <vector>

#include "garbling/gadgets/base_extension_gadget.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/layer/garbled_layer.h"
#include "circuit/layer/base_extension.h"
#include "misc/datatypes.h"
#include "misc/util.h"

using std::vector;

class GarbledBaseExtension : public GarbledLayer
{
    BaseExtensionGadget *m_base_extension_gadget;
    vector<size_t> m_extra_moduli_indices;

public:
    GarbledBaseExtension(Layer *layer_ptr, GarbledCircuitInterface *gc)
        : GarbledLayer{layer_ptr, gc}
    {
        m_base_extension_gadget = new BaseExtensionGadget{
            m_gc,
            m_layer->get_input_size(),
            m_gc->get_crt_base(),
            dynamic_cast<BaseExtension *>(m_layer)->get_extra_moduli()};
        for (auto modulus : dynamic_cast<BaseExtension *>(m_layer)->get_extra_moduli())
        {
            const auto modulus_index = std::find(m_gc->get_crt_base().begin(), m_gc->get_crt_base().end(), modulus) -
                                       m_gc->get_crt_base().begin();
            m_extra_moduli_indices.push_back(modulus_index);
        }
    }

    virtual ~GarbledBaseExtension()
    {
        delete m_base_extension_gadget;
    }

    void garble(vector<LabelTensor *> *in_label) override
    {
        // Reserve ouput_label
        size_t crt_base_size = m_gc->get_crt_base().size();
        dim_t output_dims = in_label->at(0)->get_dims();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            auto modulus = m_gc->get_crt_base().at(i);
            auto output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        // Set the selected residue to 0
        for (size_t input_idx = 0; input_idx < m_layer->get_input_size(); ++input_idx)
        {
            for (const auto extra_modulus_index : m_extra_moduli_indices)
            {
                in_label->at(extra_modulus_index)->set_label(m_gc->get_base_label().at(extra_modulus_index)->get_label(input_idx), input_idx);
            }
        }

        m_base_extension_gadget->garble(in_label, m_out_label);
    }

    vector<LabelTensor *> *cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                                        int nr_threads) override
    {
        // Reserve ouput_label
        free_out_label();
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            dim_t output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        // Set the selected residue to 0
        for (size_t input_idx = 0; input_idx < m_layer->get_input_size(); ++input_idx)
        {
            for (const auto extra_modulus_index : m_extra_moduli_indices)
            {
                encoded_inputs->at(extra_modulus_index)->set_label(m_gc->get_base_label().at(extra_modulus_index)->get_label(input_idx), input_idx);
            }
        }

        m_base_extension_gadget->cpu_evaluate(encoded_inputs, m_out_label, nr_threads);
        return m_out_label;
    }

    void cuda_move() override
    {
        throw "Not yet implemented!";
    }

    void cuda_evaluate(crt_val_t **dev_in_label) override
    {
        throw "Not yet implemented!";
    }

    void cuda_move_output() override
    {
        throw "Not yet implemented!";
    }

    crt_val_t **get_dev_out_label() override
    {
        return m_base_extension_gadget->get_dev_out_label();
    }
};

#endif
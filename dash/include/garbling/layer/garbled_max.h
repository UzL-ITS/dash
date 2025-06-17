#ifndef GARBLED_MAX_H
#define GARBLED_MAX_H

// #include <cstdlib>
#include <vector>

#include "garbling/garbled_circuit_interface.h"
#include "garbling/layer/garbled_layer.h"
#include "garbling/gadgets/max_gadget.h"

using std::vector;

/**
 * This is just for testing purposes.
 * @author Felix Maurer
 */
class GarbledMax : public GarbledLayer
{
public:
    GarbledMax(Layer *layer_ptr, GarbledCircuitInterface *gc)
        : GarbledLayer{layer_ptr, gc}
    {
    }

    ~GarbledMax()
    {
        free_out_label();
    }

    void garble(vector<LabelTensor *> *in_label) override
    {
        // This test layer only works for input size 2.
        // Assert that ech entry of in_label has size 2.
        for(auto label : *in_label){
            assert(label->get_nr_label() == 2);
        }

        const size_t crt_base_size = m_gc->get_crt_base().size();
        m_out_label->resize(crt_base_size);

        m_max_gadget = new MaxGadget(m_gc, 2, m_gc->get_crt_base());

        vector<LabelTensor *> in_label1{};
        vector<LabelTensor *> in_label2{};
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            // Prepare output label tensor
            const dim_t out_dims = m_layer->get_output_dims();
            const crt_val_t out_modulus = m_max_gadget->get_out_moduli().at(i);
            m_out_label->at(i) = new LabelTensor{out_modulus, out_dims};

            // Prepare input label tensors
            in_label1.push_back(new LabelTensor(in_label->at(i)->get_label(0)));
            in_label2.push_back(new LabelTensor(in_label->at(i)->get_label(1)));
        }
        // Garble the singular output label tensor
        m_max_gadget->garble(&in_label1, &in_label2, m_out_label);

        // Clean up
        for (auto label : in_label1)
        {
            delete label;
        }
        for (auto label : in_label2)
        {
            delete label;
        }
    }

    vector<LabelTensor *> *cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                                        int nr_threads) override
    {
        // This test layer only works for input size 2
        // Assert that ech entry of in_label has size 2.
        for(auto label : *encoded_inputs){
            assert(label->get_nr_label() == 2);
        }

        const size_t crt_base_size = m_gc->get_crt_base().size();

        vector<LabelTensor *> encoded_inputs1{};
        vector<LabelTensor *> encoded_inputs2{};
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            // Prepare output label tensor
            const dim_t out_dims = m_layer->get_output_dims();
            const crt_val_t out_modulus = m_max_gadget->get_out_moduli().at(i);
            m_out_label->at(i) = new LabelTensor{out_modulus, out_dims};

            // Prepare input label tensors
            encoded_inputs1.push_back(new LabelTensor(encoded_inputs->at(i)->get_label(0)));
            encoded_inputs2.push_back(new LabelTensor(encoded_inputs->at(i)->get_label(1)));
        }
        // Garble the singular output label tensor
        m_max_gadget->cpu_evaluate(&encoded_inputs1, &encoded_inputs2, m_out_label, nr_threads);

        // Clean up
        for (auto label : encoded_inputs1)
        {
            delete label;
        }
        for (auto label : encoded_inputs2)
        {
            delete label;
        }

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
        throw "Not yet implemented!";
    }

private:
    MaxGadget *m_max_gadget;
};

#endif
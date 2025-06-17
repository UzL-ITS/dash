#ifndef GARBLED_RESCALE_H
#define GARBLED_RESCALE_H

#include <vector>

#include "circuit/layer/rescale.h"
#include "garbling/gadgets/rescale_gadget.h"
#include "garbling/layer/garbled_layer.h"
#include "misc/datatypes.h"
#include "misc/util.h"

using std::vector;

class GarbledRescale : public GarbledLayer
{
    Rescale *m_rescale;
    // One rescale gadget for each rescale iteration l
    vector<RescaleGadget *> m_rescale_gadgets;

public:
    GarbledRescale(Layer *layer_ptr, GarbledCircuitInterface *gc)
        : GarbledLayer{layer_ptr, gc}
    {
        m_rescale = dynamic_cast<Rescale *>(m_layer);
        size_t input_size = m_layer->get_input_size();

        if (m_rescale->get_use_sign_base_extension())
        {
            const auto l = m_rescale->get_l();

            for (int i = 0; i < l; ++i)
            {
                m_rescale_gadgets.push_back(new RescaleGadget{gc, input_size, true, {2}});
            }
        }
        else
        {
            const auto scaling_factors = m_rescale->get_s();

            m_rescale_gadgets.push_back(new RescaleGadget{gc, input_size, false, scaling_factors});
        }
    }

    virtual ~GarbledRescale()
    {
        for (auto &gadget : m_rescale_gadgets)
        {
            delete gadget;
        }
    }

    void garble(vector<LabelTensor *> *in_label) override
    {
        // Reserve output_label
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t j = 0; j < crt_base_size; ++j)
        {
            auto modulus = m_gc->get_crt_base().at(j);
            auto label = new LabelTensor{modulus, m_layer->get_output_dims()};
            m_out_label->push_back(label);
        }

        m_rescale_gadgets.at(0)->garble(in_label, m_out_label);

        // non-sign base extension only needs one iteration
        if (m_rescale->get_use_sign_base_extension())
        {
            for (int i = 1; i < m_rescale->get_l(); ++i)
            {
                m_rescale_gadgets.at(i)->garble(m_out_label, m_out_label);
            }
        }
    }

    vector<LabelTensor *> *cpu_evaluate(vector<LabelTensor *> *encoded_inputs,
                                        int nr_threads) override
    {
        // Reserve ouput_label
        free_out_label();
        dim_t output_dims = m_layer->get_output_dims();
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        m_rescale_gadgets.at(0)->cpu_evaluate(encoded_inputs, m_out_label,
                                              nr_threads);

        // non-sign base extension only needs one iteration
        if (m_rescale->get_use_sign_base_extension())
        {
            for (int i = 1; i < m_rescale->get_l(); ++i)
            {
                m_rescale_gadgets.at(i)->cpu_evaluate(m_out_label, m_out_label,
                                                      nr_threads);
            }
        }

        return m_out_label;
    }

#ifndef SGX
    void cuda_move() override
    {
        for (auto &gadget : m_rescale_gadgets)
        {
            gadget->cuda_move();
        }
    }

    void cuda_evaluate(crt_val_t **dev_in_label) override
    {
        free_out_label();
        auto r_gadget = m_rescale_gadgets.at(0);
        crt_val_t **dev_out_label = r_gadget->cuda_evaluate(dev_in_label);
        for (int i = 1; i < m_rescale->get_l(); ++i)
        {
            r_gadget = m_rescale_gadgets.at(i);
            dev_out_label = r_gadget->cuda_evaluate(dev_out_label);
        }
    }

    void cuda_move_output() override
    {
        // Reserve ouput_label
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            dim_t output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        m_rescale_gadgets.back()->cuda_move_output(m_out_label);
    }
#else
    void cuda_move() override
    {
        for (auto &gadget : m_rescale_gadgets)
        {
            gadget->cuda_move();
        }
    }

    void cuda_evaluate(crt_val_t **dev_in_label) override
    {
        free_out_label();
        auto r_gadget = m_rescale_gadgets.at(0);
        crt_val_t **dev_out_label = r_gadget->cuda_evaluate(dev_in_label);
        for (int i = 1; i < m_rescale->get_l(); ++i)
        {
            r_gadget = m_rescale_gadgets.at(i);
            dev_out_label = r_gadget->cuda_evaluate(dev_out_label);
        }
    }

    void cuda_move_output() override
    {
        // Reserve ouput_label
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            crt_val_t modulus = m_gc->get_crt_base().at(i);
            dim_t output_dims = m_layer->get_output_dims();
            auto label = new LabelTensor{modulus, output_dims};
            m_out_label->push_back(label);
        }

        m_rescale_gadgets.back()->cuda_move_output(m_out_label);
    }
#endif
    crt_val_t **get_dev_out_label() override
    {
        return m_rescale_gadgets.back()->get_dev_out_label();
    }
};

#endif
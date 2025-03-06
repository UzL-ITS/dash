#ifndef GARBLED_CIRCUIT_H
#define GARBLED_CIRCUIT_H

#include <cassert>
#include <cstdlib>
#include <vector>

#include "circuit/circuit.h"
#include "circuit/layer/layer.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/label_tensor.h"
#include "garbling/layer/garbled_conv2d.h"
#include "garbling/layer/garbled_dense.h"
#include "garbling/layer/garbled_mixed_mod_mult.h"
#include "garbling/layer/garbled_mult.h"
#include "garbling/layer/garbled_projection.h"
#include "garbling/layer/garbled_relu.h"
#include "garbling/layer/garbled_rescale.h"
#include "garbling/layer/garbled_sign.h"
#include "garbling/layer/garbled_max.h"
#include "garbling/layer/garbled_maxpool2d.h"
#include "garbling/layer/garbled_base_extension.h"
#include "misc/datatypes.h"

#ifdef BENCHMARK
#include <chrono>
#include <map>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
#endif

using std::vector;

class GarbledCircuit : public GarbledCircuitInterface
{
#ifdef BENCHMARK
    std::map<std::string, double> m_garbling_times;
#endif

public:
    GarbledCircuit(Circuit *circuit, int cpm_size, crt_val_t max_modulus = 0,
                   bool garble_me = true)
        : GarbledCircuitInterface{circuit, cpm_size, max_modulus}
    {
        assert(circuit->get_layer().size() > 0 &&
               "Circuit must have at least one layer");
        if (garble_me)
        {
            garble();
            gen_decoding_information();
        }
    }

    GarbledCircuit(Circuit *circuit, int cpm_size,
                   const vector<mrs_val_t> &mrs_base, crt_val_t max_modulus = 0,
                   bool garble_me = true)
        : GarbledCircuitInterface{circuit, cpm_size, mrs_base, max_modulus}
    {
        assert(circuit->get_layer().size() > 0 &&
               "Circuit must have at least one layer");
        if (garble_me)
        {
            garble();
            gen_decoding_information();
        }
    }

    GarbledCircuit(Circuit *circuit, int cpm_size, float mrs_accuracy,
                   crt_val_t max_modulus = 0, bool garble_me = true)
        : GarbledCircuitInterface{circuit, cpm_size, mrs_accuracy,
                                  max_modulus}
    {
        assert(circuit->get_layer().size() > 0 &&
               "Circuit must have at least one layer");
        if (garble_me)
        {
            garble();
            gen_decoding_information();
        }
    }

    GarbledCircuit(Circuit *circuit, const vector<crt_val_t> &crt_base,
                   crt_val_t max_modulus = 0, bool garble_me = true)
        : GarbledCircuitInterface{circuit, crt_base, max_modulus}
    {
        assert(circuit->get_layer().size() > 0 &&
               "Circuit must have at least one layer");
        if (garble_me)
        {
            garble();
            gen_decoding_information();
        }
    }

    GarbledCircuit(Circuit *circuit, const vector<crt_val_t> &crt_base,
                   const vector<mrs_val_t> &mrs_base, crt_val_t max_modulus = 0,
                   bool garble_me = true)
        : GarbledCircuitInterface{circuit, crt_base, mrs_base, max_modulus}
    {
        assert(circuit->get_layer().size() > 0 &&
               "Circuit must have at least one layer");
        if (garble_me)
        {
            garble();
            gen_decoding_information();
        }
    }

#ifdef BENCHMARK
    std::map<std::string, double> get_garbling_times()
    {
        return m_garbling_times;
    }
#endif

private:
    /*
     * @brief Garbles the circuit from input to output layer.
     *
     * Uses the base label of the circuit as input label in the first layer of
     * the garbled circuit. Then, the input labels of the layers are
     * successively given by the output labels of the previous layer.
     *
     * @param circuit Plaintext circuit.
     * @param crt_base CRT base used for garbling.
     */
    void garble() override
    {
        init_garbling();
        for (size_t i = 0; i < m_circuit->get_layer().size(); ++i)
        {
            vector<LabelTensor *> *in_label;
            // If not first layer, use output labels of previous layer.
            if (i > 0)
            {
                in_label = m_garbled_layer.back()->get_out_label();
            }
            // First layer gets base label as input label.
            else
            {
                in_label = &m_base_label;
            }

            // Garble circuit layer by layer.
            Layer *layer = m_circuit->get_layer().at(i);

            // Handle flatten layer separately
            if (layer->get_type() == Layer::flatten)
            {
                for (auto l : *in_label)
                {
                    l->flatten();
                }
                i++;
                if (i >= m_circuit->get_layer().size())
                    return;
                layer = m_circuit->get_layer().at(i);
            }

            if (layer->get_type() == Layer::dense)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto garbled_dense{new GarbledDense{layer, this}};
                garbled_dense->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("dense") == m_garbling_times.end())
                {
                    m_garbling_times["dense"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["dense"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(garbled_dense);
            }
            else if (layer->get_type() == Layer::conv2d)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto garbled_conv2d{new GarbledConv2d{layer, this}};
                garbled_conv2d->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("conv2d") == m_garbling_times.end())
                {
                    m_garbling_times["conv2d"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["conv2d"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(garbled_conv2d);
            }
            else if (layer->get_type() == Layer::projection)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto garbled_proj{new GarbledProjection{layer, this}};
                garbled_proj->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("projection") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["projection"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["projection"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(garbled_proj);
            }
            else if (layer->get_type() == Layer::mult_layer)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto garbled_mult_layer{new GarbledMult{layer, this}};
                garbled_mult_layer->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("mult_layer") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["mult_layer"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["mult_layer"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(garbled_mult_layer);
            }
            else if (layer->get_type() == Layer::mixed_mod_mult_layer)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_mixed_mod_mult{new GarbledMixedModMult{layer, this}};
                g_mixed_mod_mult->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("mixed_mod_mult_layer") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["mixed_mod_mult_layer"] =
                        ms_double.count();
                }
                else
                {
                    m_garbling_times["mixed_mod_mult_layer"] +=
                        ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_mixed_mod_mult);
            }
            else if (layer->get_type() == Layer::approx_relu)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_approx_relu{new GarbledRelu{layer, this}};
                g_approx_relu->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("approx_relu") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["approx_relu"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["approx_relu"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_approx_relu);
            }
            else if (layer->get_type() == Layer::sign)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_sign{new GarbledSign{layer, this}};
                g_sign->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("sign") == m_garbling_times.end())
                {
                    m_garbling_times["sign"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["sign"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_sign);
            }
            else if (layer->get_type() == Layer::rescale)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_rescale{new GarbledRescale{layer, this}};
                g_rescale->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("rescale") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["rescale"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["rescale"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_rescale);
            }
            else if (layer->get_type() == Layer::max)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_max{new GarbledMax{layer, this}};
                g_max->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("max") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["max"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["max"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_max);
            }
            else if (layer->get_type() == Layer::max_pool)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_maxpool{new GarbledMaxPool2d{layer, this}};
                g_maxpool->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("max_pool") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["max_pool"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["max_pool"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_maxpool);
            }
            else if (layer->get_type() == Layer::base_extension)
            {
#ifdef BENCHMARK
                auto t1 = high_resolution_clock::now();
#endif
                auto g_be{new GarbledBaseExtension{layer, this}};
                g_be->garble(in_label);
#ifdef BENCHMARK
                auto t2 = high_resolution_clock::now();
                duration<double, std::milli> ms_double = t2 - t1;
                if (m_garbling_times.find("base_extension") ==
                    m_garbling_times.end())
                {
                    m_garbling_times["base_extension"] = ms_double.count();
                }
                else
                {
                    m_garbling_times["base_extension"] += ms_double.count();
                }
#endif
                m_garbled_layer.push_back(g_be);
            }
            else
            {
                throw std::runtime_error("Garbling failed: Unknown layer type");
            }
        }
    }
};

#endif
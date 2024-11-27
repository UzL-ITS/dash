#ifndef GARBLED_LAYER_H
#define GARBLED_LAYER_H

#include <numeric>
#include <vector>

#include "garbling/label_tensor.h"
#include "circuit/layer/layer.h"

using std::vector;

// Forward declaration of Interface, we can not include it here, otherwise
// we will get circular dependency.
class GarbledCircuitInterface;

class GarbledLayer {
   protected:
    Layer* m_layer;
    GarbledCircuitInterface* m_gc;

    // label tensor per crt modulus
    vector<LabelTensor*>* m_in_label;
    vector<LabelTensor*>* m_out_label;

   public:
    GarbledLayer(Layer* layer, GarbledCircuitInterface* gc)
        : m_layer{layer},
          m_gc{gc},
          m_in_label{nullptr},
          m_out_label{new vector<LabelTensor*>{}} {}

    virtual ~GarbledLayer() {
        for (auto& label : *m_out_label) {
            delete label;
        }
        delete m_out_label;
    };

    virtual void garble(vector<LabelTensor*>* in_label) = 0;

    /**
     * @brief Evaluate layer on CPU.
     *
     */
    virtual vector<LabelTensor*>* cpu_evaluate(vector<LabelTensor*>* g_inputs,
                                               int nr_threads) = 0;

    /**
     * @brief Move layer members to GPU and allocate needed GPU memory
     *
     */
    virtual void cuda_move() = 0;

    /**
     * @brief Evaluate layer with given input on GPU.
     *
     * First layer needs to be moved to GPU.
     *
     * @param dev_in_label
     */
    virtual void cuda_evaluate(crt_val_t** dev_in_label) = 0;

    /**
     * @brief Move outputs of layer to CPU
     *
     */
    virtual void cuda_move_output() = 0;

    /**
     * @brief Get pointer to ouput labels on GPU
     *
     * Needed by garbled circuit, to evaluate circuit layer after layer on GPU.
     * Sucessively each layer gets pointer to output labels of previous layer.
     *
     * @return crt_val_t**
     */
    virtual crt_val_t** get_dev_out_label() = 0;

    Layer* get_layer() { return m_layer; }
    GarbledCircuitInterface* get_gc() { return m_gc; }
    vector<LabelTensor*>* get_in_label() { return m_in_label; }
    vector<LabelTensor*>* get_out_label() { return m_out_label; }

    /**
     * @brief Free ouput label on CPU
     *
     * Needed before evaluating an initialized layer. Otherwise there would be
     * a memory leak from the garbling process.
     *
     */
    virtual void free_out_label() {
        for (auto& label : *m_out_label) {
            delete label;
        }
        m_out_label->clear();
    }
};

#endif
#ifndef MAX_GADGET_H
#define MAX_GADGET_H

#include <cstdlib>
#include <cstring>
#include <vector>

#include "garbling/garbled_circuit_interface.h"
#include "garbling/gadgets/relu_gadget.h"
#include "misc/datatypes.h"
#include "misc/misc.h"

using std::vector;

/**
 * @class MaxGadget
 * @brief The MaxGadget class represents a gadget for computing the maximum of two inputs.
 * @author Felix Maurer
 *
 * This class provides methods for garbling and evaluating the maximum gadget, as well as
 * CUDA-specific methods for moving data and evaluating the gadget on a CUDA device.
 *
 * The maximum gadget is implemented based on the Ball et al. (2019) paper, where max(x, y) = x + relu(y - x).
 *
 * The class takes a GarbledCircuitInterface object, the number of inputs, and the output moduli as parameters
 * in the constructor. It uses a ReluGadget object internally to perform the relu operation.
 *
 * The garble() method garbles the maximum gadget by performing the subtraction, relu, and addition operations
 * on the input label tensors. The cpu_evaluate() method evaluates the maximum gadget on the CPU by performing
 * the same operations. The cuda_move(), cuda_evaluate(), and cuda_move_output() methods are placeholders for
 * CUDA-specific functionality and are not yet implemented.
 *
 * The class also provides getter methods for retrieving the output moduli and the device output label.
 */
class MaxGadget
{

public:
    MaxGadget(GarbledCircuitInterface *gc, const size_t nr_inputs, const vector<crt_val_t> out_moduli)
        : m_gc{gc},
          M_NR_INPUTS{nr_inputs},
          M_OUT_MODULI{out_moduli},
          m_relu_gadget{new ReluGadget(m_gc, 1)} // we only input one value to the relu gadget
    {
    }

    ~MaxGadget()
    {
        delete m_relu_gadget;
    }

    void garble(vector<LabelTensor *> *in_label1,
                vector<LabelTensor *> *in_label2,
                vector<LabelTensor *> *out_label)
    {
        const vector<crt_val_t> crt_base = m_gc->get_crt_base();
        const size_t crt_base_size = crt_base.size();

        // Assert dimension of input and output labels
        assert(in_label1->size() == crt_base_size);
        assert(in_label2->size() == crt_base_size);
        assert(out_label->size() == crt_base_size);
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            assert(in_label1->at(crt_idx)->get_dims() == dim_t{1});
            assert(in_label2->at(crt_idx)->get_dims() == dim_t{1});
            assert(out_label->at(crt_idx)->get_dims() == dim_t{1});
        }

        // Ball et al. (2019): max(x,y) = x + relu(y-x)
        // 1. Subtraction
        vector<LabelTensor *> out_label_subtraction{};
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            // 1.1 Init new LabelTensor for subtraction output
            out_label_subtraction.push_back(new LabelTensor{crt_base.at(crt_idx)});

            // 1.2 Subtract on slices
            LabelSlice subtraction_slice = in_label2->at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
            subtraction_slice -= in_label1->at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);

            out_label_subtraction.at(crt_idx)->set_label(subtraction_slice, SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
        }
        // 2. Relu
        // 2.1 Init new LabelTensors for Relu output
        vector<LabelTensor *> out_label_relu{};
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            out_label_relu.push_back(new LabelTensor{crt_base.at(i)});
        }
        // 2.3 Apply Relu gadget
        m_relu_gadget->garble(&out_label_subtraction, &out_label_relu);

        // 3. Addition
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            // 3.1 Add on slices
            LabelSlice addition_slice = in_label1->at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
            addition_slice += out_label_relu.at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);

            // 3.2 Update out_label
            out_label->at(crt_idx)->set_label(addition_slice, SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
        }

        for (auto label : out_label_subtraction)
        {
            delete label;
        }

        for (auto label : out_label_relu)
        {
            delete label;
        }
    }

    void cpu_evaluate(vector<LabelTensor *> *encoded_inputs1,
                      vector<LabelTensor *> *encoded_inputs2,
                      vector<LabelTensor *> *out_label,
                      int nr_threads)
    {
        const vector<crt_val_t> crt_base = m_gc->get_crt_base();
        const size_t crt_base_size = crt_base.size();

        // Assert dimension of input and output labels
        assert(encoded_inputs1->size() == crt_base_size);
        assert(encoded_inputs2->size() == crt_base_size);
        assert(out_label->size() == crt_base_size);
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            assert(encoded_inputs1->at(crt_idx)->get_dims() == dim_t{1});
            assert(encoded_inputs2->at(crt_idx)->get_dims() == dim_t{1});
            assert(out_label->at(crt_idx)->get_dims() == dim_t{1});
        }

        // Ball et al. (2019): max(x,y) = x + relu(y-x)
        // 1. Subtraction
        // 1.1 Init new LabelTensor for subtraction output
        vector<LabelTensor *> out_label_subtraction{};
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            out_label_subtraction.push_back(new LabelTensor{crt_base.at(crt_idx)});
            // 1.2 Subtract on slices
            LabelSlice subtraction_slice = encoded_inputs2->at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
            subtraction_slice -= encoded_inputs1->at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);

            out_label_subtraction.at(crt_idx)->set_label(subtraction_slice, SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
        }
        // 2. Relu
        // 2.1 Init new LabelTensors for Relu output
        vector<LabelTensor *> out_label_relu{};
        for (size_t i = 0; i < crt_base_size; ++i)
        {
            out_label_relu.push_back(new LabelTensor{crt_base.at(i)});
        }
        // 2.3 Apply Relu gadget
        m_relu_gadget->cpu_evaluate(&out_label_subtraction, &out_label_relu, M_NR_INPUTS);

        // 3. Addition
        for (size_t crt_idx = 0; crt_idx < crt_base_size; ++crt_idx)
        {
            // 3.1 Add on slices
            LabelSlice addition_slice = encoded_inputs1->at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
            addition_slice += out_label_relu.at(crt_idx)->get_label(SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);

            // 3.2 Update out_label
            out_label->at(crt_idx)->set_label(addition_slice, SINGLE_ENTRY_TENSOR_DEFAULT_INDEX);
        }

        for (auto label : out_label_subtraction)
        {
            delete label;
        }

        for (auto label : out_label_relu)
        {
            delete label;
        }
    }

    void cuda_move()
    {
        throw "Not yet implemented!";
    }

    void cuda_evaluate(crt_val_t **dev_in_label)
    {
        throw "Not yet implemented!";
    }

    void cuda_move_output()
    {
        throw "Not yet implemented!";
    }

    crt_val_t **get_dev_out_label()
    {
        throw "Not yet implemented!";
    }

    vector<crt_val_t> get_out_moduli() const
    {
        return M_OUT_MODULI;
    }

private:
    GarbledCircuitInterface *m_gc;

    const size_t M_NR_INPUTS;
    const vector<crt_val_t> M_OUT_MODULI;

    ReluGadget *m_relu_gadget{};

    /**
     * Use this index to set label of 1 x 1 ... x 1 LabelTensors
     */
    static constexpr size_t SINGLE_ENTRY_TENSOR_DEFAULT_INDEX = 0;
};

#endif // MAX_GADGET_H
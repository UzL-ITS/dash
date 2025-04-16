// This is a non-sgx main for testing purposes.
#include <random>
#include <vector>
#include <iostream>

#include "circuit/circuit.h"
#include "circuit/layer/conv2d.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/flatten.h"
#include "circuit/layer/layer.h"
#include "circuit/layer/max_pool2d.h"
#include "circuit/layer/mixed_mod_mult_layer.h"
#include "circuit/layer/mult_layer.h"
#include "circuit/layer/projection.h"
#include "circuit/layer/relu.h"
#include "circuit/layer/rescale.h"
#include "circuit/onnx_modelloader.h"
#include "garbling/gadgets/lookup_approx_sign.h"
#include "garbling/garbled_circuit.h"
#include "garbling/label_tensor.h"
#include "garbling/layer/garbled_dense.h"
#include "misc/cuda_util.h"
#include "misc/dataloader.h"
#include "misc/misc.h"
#include "misc/util.h"

using std::vector;

int main() {
    init_cuda();

    //
    //
    // Example: Automatic inference of quantization constant
    // auto dataset = cifar10("../../data/cifar10/cifar-10-batches-bin");
    // vector<wandb_t> mean = {0.4914, 0.4822, 0.4465};
    // vector<wandb_t> std = {0.247, 0.243, 0.261};
    // normalize(&dataset, mean, std);

    auto dataset = mnist("../../data/MNIST/raw");

    // Create subset of dataset
    vector<ScalarTensor<wandb_t>> small_test_images;
    vector<unsigned long> small_test_labels;
    for (int i = 0; i < 100; ++i) {
        auto tmp_image{dataset.testing_images.at(i)};
        auto tmp_label{dataset.testing_labels.at(i)};
        small_test_images.push_back(tmp_image);
        small_test_labels.push_back(tmp_label);
    }

    int target_crt_base_size = 7;

    std::string model_path = "../../models/MODEL_A.onnx";
    Circuit* circuit =
        load_onnx_model(model_path, QuantizationMethod::SimpleQuant);
    circuit->print();

    int nr_samples = 10;
    circuit->optimize_quantization(target_crt_base_size, dataset.testing_images,
                                   0.1, 0.001, 0.0001, nr_samples);
    auto q_const = circuit->get_q_const();

    std::cout << "q_const: " << q_const << std::endl;

    // auto quantized_inputs =
    //     quantize(dataset.testing_images, QuantizationMethod::ScaleQuant, QL);
    auto quantized_inputs = quantize(dataset.testing_images,
    QuantizationMethod::SimpleQuant, q_const);

    auto plain_test_acc =
        circuit->plain_test(dataset.testing_images, dataset.testing_labels);
    printf("plain_test_acc: %f\n", plain_test_acc);

    auto plain_q_test_acc =
        circuit->plain_q_test(quantized_inputs, dataset.testing_labels);
    printf("plain_q_test_acc: %f\n", plain_q_test_acc);

    auto gc = new GarbledCircuit(circuit, target_crt_base_size, (float)100.0);
    gc->cuda_move();
    auto g_inputs{gc->garble_inputs(quantized_inputs.at(0))};
    auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};

    gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{gc->cuda_move_outputs()};

    auto outputs{gc->decode_outputs(g_outputs)};

    std::cout << "output_label: " << outputs.argmax() << std::endl;

    std::cout << "\noutputs: " << std::endl;
    outputs.print();

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    gc->cuda_free_inputs(g_dev_inputs);

    delete circuit;
    delete gc;
    //
    //
    //
}

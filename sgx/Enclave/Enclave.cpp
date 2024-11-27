/*
 * Copyright (C) 2011-2021 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#define SGX

#include "Enclave_t.h"
#include "circuit/circuit.h"
#include "circuit/layer/conv2d.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/layer.h"
#include "circuit/layer/relu.h"
#include "circuit/onnx_modelloader.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit.h"
#include "garbling/label_tensor.h"
#include "misc/dataloader.h"
#include "misc/datatypes.h"
#include "misc/enclave_functions.h"
#include "onnx.proto3.pb.h"

using std::vector;

void print_label(LabelTensor* l) {
    for (int i = 0; i < l->get_nr_label(); ++i) {
        for (int j = 0; j < l->get_nr_comps(); ++j) {
            printf("%d ", l->get_components()[i * l->get_nr_comps() + j]);
        }
        printf("\n");
    }
    printf("\n");
}

//
//
// Example-Code: Dense Layer
// void ecall_ann_infer(void* inputs_tmp) {
//     // Cast inputs
//     auto inputs_ptr = static_cast<ScalarTensor<q_val_t>*>(inputs_tmp);
//     auto inputs = *inputs_ptr;

//     ScalarTensor<wandb_t> weights{{10,20,30,40,50,60}, dim_t{3,2}};
//     ScalarTensor<wandb_t> biases{{50,60,70}, dim_t{3}};

//     auto circuit = new Circuit{new Dense{weights, biases,
//     QuantizationMethod::SimpleQuant, 10}}; auto gc = new
//     GarbledCircuit(circuit, 8); gc->cuda_move(); auto
//     g_inputs{gc->garble_inputs(inputs)}; auto
//     g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//     gc->cuda_evaluate(g_dev_inputs);
//     auto g_outputs{gc->cuda_move_outputs()};
//     auto outputs{gc->decode_outputs(g_outputs)};

//     printf("outputs:\n");
//     outputs.print();

//     // clean up
//     for(auto label : *g_inputs) {
//         delete label;
//     }
//     delete g_inputs;

//     gc->cuda_free_inputs(g_dev_inputs);
//     delete circuit;
//     delete gc;

//     printf("Finished ANN inference\n");
// }
//
//
//

//
//
// Example-Code: Conv2D Layer
void ecall_ann_infer(void* inputs_tmp) {
    // Cast inputs
    auto inputs_ptr = static_cast<ScalarTensor<q_val_t>*>(inputs_tmp);
    auto inputs = *inputs_ptr;

    // Conv parameters
    int input_width = 3;
    int input_height = 3;
    int input_size = input_height * input_width;
    int channel = 1;

    int filter = 1;
    int filter_width = 1;
    int filter_height = 1;
    int filter_size = filter_width * filter_height;

    int stride_width = 1;
    int stride_height = 1;

    // Generate example data
    int size = filter_size * channel * filter * sizeof(wandb_t);
    wandb_t* weights = (wandb_t*)malloc(size);
    wandb_t* bias = (wandb_t*)malloc(filter * sizeof(wandb_t));

    for (int i = 0; i < filter_size * channel * filter; ++i) {
        weights[i] = 10;  //(i*10)%100;
    }

    for (int i = 0; i < filter; ++i) {
        bias[i] = 0;  //(i*10)%100;
    }

    dim_t weights_dims{filter_width, filter_height, channel, filter};
    ScalarTensor<wandb_t> weights_t{weights, weights_dims};

    dim_t bias_dims{filter};
    ScalarTensor<wandb_t> bias_t{bias, bias_dims};

    auto circuit = new Circuit{new Conv2d(
        weights_t, bias_t, input_width, input_height, channel, filter,
        filter_width, filter_height, stride_width, stride_height,
        QuantizationMethod::SimpleQuant, 10)};

    auto gc = new GarbledCircuit(circuit, 8);
    gc->cuda_move();
    auto g_inputs{gc->garble_inputs(inputs)};
    auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{gc->cuda_move_outputs()};
    auto outputs{gc->decode_outputs(g_outputs)};

    printf("output_dims:\n");
    for (auto dim : outputs.get_dims()) {
        printf("%ld ", dim);
    }
    printf("\n");

    printf("outputs:\n");
    outputs.print();

    // clean up
    free(weights);
    free(bias);

    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    gc->cuda_free_inputs(g_dev_inputs);

    delete circuit;
    delete gc;

    printf("Finished ANN inference\n");
}
//
//
//

//
//
// Example-Code: Relu Layer / Sign Layer
// void ecall_ann_infer(void* inputs_tmp) {
//     // Cast inputs
//     auto inputs_ptr = static_cast<ScalarTensor<q_val_t>*>(inputs_tmp);
//     auto inputs = *inputs_ptr;

//     //auto circuit = new Circuit{new Sign{inputs.get_dims()}};
//     auto circuit = new Circuit{new Relu{inputs.get_dims()}};

//     auto gc = new GarbledCircuit(circuit, 9, (float)100.0);
//     gc->cuda_move();
//     auto g_inputs{gc->garble_inputs(inputs)};
//     auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//     // auto g_outputs{gc->cpu_evaluate(g_inputs)};
//     gc->cuda_evaluate(g_dev_inputs);
//     auto g_outputs{gc->cuda_move_outputs()};
//     auto outputs{gc->decode_outputs(g_outputs)};
//     printf("outputs:\n");
//     outputs.print();

//     //clean up
//     for (auto label : *g_inputs) {
//         delete label;
//     }
//     delete g_inputs;
//     gc->cuda_free_inputs(g_dev_inputs);

//     delete circuit;
//     delete gc;

//     printf("Finished ANN inference\n");
// }
//
//
//

//
//
// Example-Code: Rescaling Layer
// void ecall_ann_infer(void* inputs_tmp) {
//     // Cast inputs
//     auto inputs_ptr = static_cast<ScalarTensor<q_val_t>*>(inputs_tmp);
//     auto inputs = *inputs_ptr;

//     // auto circuit = new Circuit{new Sign{inputs.get_dims()}};
//     auto circuit = new Circuit{new Rescale{2, inputs.get_dims()}};

//     vector<crt_val_t> crt_base{2, 3, 5};
//     vector<mrs_val_t> mrs_base{26, 3};

//     // auto gc = new GarbledCircuit(circuit, 3, (float)100.0);
//     auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);
//     gc->cuda_move();
//     auto g_inputs{gc->garble_inputs(inputs)};
//     auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//     // auto g_outputs{gc->cpu_evaluate(g_inputs)};
//     gc->cuda_evaluate(g_dev_inputs);
//     auto g_outputs{gc->cuda_move_outputs()};
//     auto outputs{gc->decode_outputs(g_outputs)};
//     printf("outputs:\n");
//     outputs.print();

//     // clean up
//     for (auto label : *g_inputs) {
//         delete label;
//     }
//     delete g_inputs;
//     gc->cuda_free_inputs(g_dev_inputs);

//     delete circuit;
//     delete gc;

//     printf("Finished ANN inference\n");
// }
//
//
//

//
//
// Example-Code: End-To-End
// void ecall_ann_infer(void* inputs_tmp) {
//     // Dataset
//     auto dataset = static_cast<dataset_tensor*>(inputs_tmp);
//     auto inputs = dataset->testing_images.at(0);

//     int target_crt_base_size = 9;

//     // Load and optimize model
//     std::string model_path = "../models/trained/MODEL_A.onnx";
//     auto circuit = load_onnx_model(model_path);
//     circuit->optimize_quantization(target_crt_base_size,
//                                    dataset->testing_images, 0.25, 0.01,
//                                    0.0000001, 5);
//     auto q_const = circuit->get_q_const();
//     // Circuit* circuit = load_onnx_model(model_path,
//     // QuantizationMethod::ScaleQuant);

//     auto plain_outputs = circuit->plain_eval(inputs);

//     printf("plain outputs:\n");
//     plain_outputs.print();

//     auto q_inputs = ScalarTensor<q_val_t>::quantize(
//         inputs, QuantizationMethod::SimpleQuant, q_const);
//     // auto q_inputs = ScalarTensor<q_val_t>::quantize(inputs,
//     // QuantizationMethod::ScaleQuant, QL);

//     auto plain_q_outputs = circuit->plain_q_eval(q_inputs);
//     printf("\nplain quantized outputs: \n");
//     plain_q_outputs.print();

//     for (auto l : circuit->get_layer()) {
//         printf("min: %lld\n", l->get_min_plain_q_val());
//         printf("max: %lld\n", l->get_max_plain_q_val());
//         printf("\n");
//     }

//     // you can also use the whole dataset.training_images vector
//     // int crt_base_size = circuit->infer_crt_base_size(q_inputs);

//     // printf("crt_base_size: %d\n", crt_base_size);
//     auto gc = new GarbledCircuit(circuit, target_crt_base_size, (float)100.0);
//     gc->cuda_move();
//     auto g_inputs{gc->garble_inputs(q_inputs)};
//     auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//     gc->cuda_evaluate(g_dev_inputs);
//     auto g_outputs = gc->cuda_move_outputs();
//     auto outputs{gc->decode_outputs(g_outputs)};

//     printf("output_label: %ld\n", outputs.argmax());

//     printf("\noutputs:\n");
//     outputs.print();

//     // clean up
//     for (auto label : *g_inputs) {
//         delete label;
//     }
//     delete g_inputs;
//     gc->cuda_free_inputs(g_dev_inputs);

//     delete circuit;
//     delete gc;

//     printf("Finished ANN inference\n");
// }
//
//
//
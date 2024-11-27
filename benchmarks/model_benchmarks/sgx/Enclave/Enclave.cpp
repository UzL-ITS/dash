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
#include "circuit/onnx_modelloader.h"
#include "circuit/scalar_tensor.h"
#include "crypto/cpu_aes_engine.h"
#include "garbling/garbled_circuit.h"
#include "garbling/layer/garbled_conv2d.h"
#include "garbling/layer/garbled_dense.h"
#include "garbling/layer/garbled_layer.h"
#include "misc/dataloader.h"
#include "misc/datatypes.h"
#include "misc/enclave_functions.h"
#include "onnx.proto3.pb.h"

using std::vector;

//
//
// Benchmark CPU inference
void ecall_ann_infer_cpu(void* inputs_tmp, void* labels_tmp, int nr_inputs,
                         const char* model_path, int target_crt_base_size,
                         void* relu_accs_tmp, long double* runtimes,
                         void* quantization_method) {
    printf("Bench (CPU) Model: %s\n", model_path);
    // Cast parameter
    auto inputs_ptr = static_cast<vector<ScalarTensor<wandb_t>>*>(inputs_tmp);
    auto inputs_vec = *inputs_ptr;
    auto labels_ptr = static_cast<vector<unsigned long>*>(labels_tmp);
    auto labels_vec = *labels_ptr;
    auto relu_accs_ptr = static_cast<vector<float>*>(relu_accs_tmp);
    auto relu_accs_vec = *relu_accs_ptr;
    QuantizationMethod q_method = *(QuantizationMethod*)quantization_method;
    // Prepare circuit
    Circuit* circuit;
    vector<ScalarTensor<q_val_t>> q_inputs_vec;
    if (q_method == QuantizationMethod::SimpleQuant) {
        circuit = load_onnx_model(model_path);
        vector<ScalarTensor<wandb_t>> sub_inputs(inputs_vec.begin(),
                                                 inputs_vec.begin() + 10);
        circuit->optimize_quantization(target_crt_base_size, sub_inputs, 0.2,
                                       0.01, 0.001);
        wandb_t q_const = circuit->get_q_const();
        q_inputs_vec = quantize(inputs_vec, q_method, q_const);
    } else {
        circuit = load_onnx_model(model_path, q_method, QL);
        q_inputs_vec = quantize(inputs_vec, q_method, QL);
    }

    for (int j = 0; j < relu_accs_vec.size(); j++) {
        auto relu_acc = relu_accs_vec.at(j);
        printf("Relu Accuracy: %f\n", relu_acc);
        vector<unsigned long> infered_labels;
        for (int i = 0; i < nr_inputs; ++i) {
            auto q_inputs = q_inputs_vec.at(i);

            auto gc =
                new GarbledCircuit(circuit, target_crt_base_size, relu_acc);

            void* t1;
            void* t2;

            ocall_clock(&t1);
            auto g_inputs{gc->garble_inputs(q_inputs)};
            auto g_outputs{gc->cpu_evaluate(g_inputs)};
            auto outputs{gc->decode_outputs(g_outputs)};
            ocall_clock(&t2);

            auto infered_label = outputs.argmax();
            infered_labels.push_back(infered_label);

            ocall_get_runtime(&runtimes[j * nr_inputs + i], t1, t2);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;
            delete gc;
        }
        // compute prediction-accuracy
        int correct = 0;
        for (size_t i = 0; i < infered_labels.size(); i++) {
            if (infered_labels.at(i) == labels_vec.at(i)) {
                correct++;
            }
        }
        double accuracy = (double)correct / (double)infered_labels.size();
        printf("Accuracy: %f\n", accuracy);
    }
    delete circuit;
}
//
//
//

//
//
// Benchmark GPU inference
void ecall_ann_infer_gpu(void* inputs_tmp, void* labels_tmp, int nr_inputs,
                         const char* model_path, int target_crt_base_size,
                         void* relu_accs_tmp, long double* runtimes,
                         void* quantization_method) {
    printf("Bench (GPU) Model: %s\n", model_path);
    // Cast parameter
    auto inputs_ptr = static_cast<vector<ScalarTensor<wandb_t>>*>(inputs_tmp);
    auto inputs_vec = *inputs_ptr;
    auto labels_ptr = static_cast<vector<unsigned long>*>(labels_tmp);
    auto labels_vec = *labels_ptr;
    auto relu_accs_ptr = static_cast<vector<float>*>(relu_accs_tmp);
    auto relu_accs_vec = *relu_accs_ptr;
    QuantizationMethod q_method = *(QuantizationMethod*)quantization_method;
    // Prepare circuit
    Circuit* circuit;
    vector<ScalarTensor<q_val_t>> q_inputs_vec;
    if (q_method == QuantizationMethod::SimpleQuant) {
        circuit = load_onnx_model(model_path);
        vector<ScalarTensor<wandb_t>> sub_inputs(inputs_vec.begin(),
                                                 inputs_vec.begin() + 10);
        circuit->optimize_quantization(target_crt_base_size, sub_inputs, 0.2,
                                       0.01, 0.001);
        wandb_t q_const = circuit->get_q_const();
        q_inputs_vec = quantize(inputs_vec, q_method, q_const);
    } else {
        circuit = load_onnx_model(model_path, q_method, QL);
        q_inputs_vec = quantize(inputs_vec, q_method, QL);
    }

    for (int j = 0; j < relu_accs_vec.size(); j++) {
        auto relu_acc = relu_accs_vec.at(j);
        printf("Relu Accuracy: %f\n", relu_acc);
        vector<unsigned long> infered_labels;
        for (int i = 0; i < nr_inputs; ++i) {
            auto q_inputs = q_inputs_vec.at(i);

            auto gc =
                new GarbledCircuit(circuit, target_crt_base_size, relu_acc);
            gc->cuda_move();
            void* t1;
            void* t2;

            ocall_clock(&t1);
            auto g_inputs{gc->garble_inputs(q_inputs)};
            auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
            gc->cuda_evaluate(g_dev_inputs);
            auto g_outputs{gc->cuda_move_outputs()};
            auto outputs{gc->decode_outputs(g_outputs)};
            ocall_clock(&t2);

            auto infered_label = outputs.argmax();
            infered_labels.push_back(infered_label);

            ocall_get_runtime(&runtimes[j * nr_inputs + i], t1, t2);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;
            gc->cuda_free_inputs(g_dev_inputs);
            delete gc;
        }
        // compute prediction-accuracy
        int correct = 0;
        for (size_t i = 0; i < infered_labels.size(); i++) {
            if (infered_labels.at(i) == labels_vec.at(i)) {
                correct++;
            }
        }
        double accuracy = (double)correct / (double)infered_labels.size();
        printf("Accuracy: %f\n", accuracy);
    }
    delete circuit;
}
//
//
//

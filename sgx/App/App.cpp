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

#include <stdio.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "Enclave_u.h"
#include "circuit/onnx_modelloader.h"
#include "garbling/label_tensor.h"
#include "garbling/layer/garbled_relu.h"
#include "garbling/layer/garbled_sign.h"
#include "misc/cuda_util.h"
#include "misc/dataloader.h"
#include "misc/datatypes.h"
#include "sgx_urts.h"

// Enclave ID
sgx_enclave_id_t global_eid = 0;

using std::vector;

/**
 * Initialize an SGX enclave.
 * @return error state
 */
int initialize_enclave() {
    sgx_status_t state = SGX_ERROR_UNEXPECTED;

    char enclavefile[256];
    getcwd(enclavefile, sizeof(enclavefile));
    strcat(enclavefile, "/enclave.signed.so");

    state = sgx_create_enclave(enclavefile, SGX_DEBUG_FLAG, NULL, NULL,
                               &global_eid, NULL);

    if (state != SGX_SUCCESS) {
        printf("Can not create enclave, error-code: %d\n", state);
        return -1;
    }

    return 0;
}

void ocall_print_string(const char* str) {
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    printf("%s", str);
    fflush(stdout);
}

raw_data_t ocall_read_file(const char* path) { return read_file(path); }

void ocall_cuda_print_g_inputs(crt_val_t** dev_garbled_inputs,
                               crt_val_t* crt_base, int crt_base_size,
                               int nr_inputs) {
    int nr_streams = crt_base_size;
    cudaStream_t stream[nr_streams];
    for (int i = 0; i < crt_base_size; i++) {
        cudaCheckError(cudaStreamCreate(&stream[i]));

        crt_val_t modulus = crt_base[i];
        int nr_comps = LabelTensor::get_nr_comps(modulus);

        print_garbled_inputs<<<1, 1, 0, stream[i]>>>(dev_garbled_inputs[i],
                                                     nr_inputs, nr_comps);

        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaStreamCreate(&stream[i]));
    }
}

void* ocall_alloc_array(size_t size) {
    void* array = malloc(size);
    return array;
}

void** ocall_alloc_ptr_array(size_t size) {
    void** ptr_array = (void**)malloc(size);
    return ptr_array;
}

void* ocall_cudaMalloc(size_t size) {
    void* dev_mem;
    cudaCheckError(cudaMalloc((void**)&dev_mem, size));
    return dev_mem;
}

void ocall_cudaFree(void* ptr) { cudaCheckError(cudaFree(ptr)); }

void ocall_cudaMemcpyToDevice(void* dst, void* src, size_t size) {
    cudaCheckError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    free(src);
}

void ocall_cudaMemcpyToDevicePtr(void** dst, void** src, size_t size) {
    cudaCheckError(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void* ocall_cudaMemcpyFromDevice(void* src, size_t size) {
    void* dst = malloc(size);
    cudaCheckError(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return dst;
}

void ocall_free(void* ptr) { free(ptr); }

void ocall_cuda_eval_dense(crt_val_t** dev_out_label, crt_val_t** dev_in_label,
                           q_val_t* dev_qe_weights,
                           crt_val_t** dev_qe_bias_label,
                           crt_val_t** dev_zero_label, int crt_base_size,
                           size_t M, size_t K, crt_val_t* crt_base) {
    int nr_streams = crt_base_size;
    cudaStream_t stream[nr_streams];

    for (int i = 0; i < nr_streams; ++i) {
        cudaCheckError(cudaStreamCreate(&stream[i]));
        crt_val_t modulus = crt_base[i];
        size_t N = LabelTensor::get_nr_comps(modulus);

        dim3 block(32, 32);
        dim3 grid((N + block.x) / block.x, (M + block.y) / block.y);

        MatMulModAddZero<<<grid, block, 0, stream[i]>>>(
            dev_qe_weights, dev_in_label[i], dev_out_label[i],
            dev_qe_bias_label[i], dev_zero_label[modulus], modulus, M, N, K);

        cudaCheckError(cudaStreamDestroy(stream[i]));
    }
    cudaDeviceSynchronize();
}

void ocall_cuda_eval_conv2d(crt_val_t** dev_out_label, crt_val_t** dev_in_label,
                            q_val_t* dev_qe_weights,
                            crt_val_t** dev_qe_bias_label,
                            crt_val_t** dev_zero_label, int crt_base_size,
                            crt_val_t* crt_base, size_t output_width,
                            size_t output_height, size_t input_width,
                            size_t input_height, size_t channel, size_t filter,
                            size_t filter_width, size_t filter_height,
                            size_t stride_width, size_t stride_height) {
    int nr_streams = crt_base_size;
    cudaStream_t stream[nr_streams];

    for (int i = 0; i < nr_streams; ++i) {
        cudaCheckError(cudaStreamCreate(&stream[i]));

        crt_val_t modulus = crt_base[i];
        int nr_comps = LabelTensor::get_nr_comps(modulus);

        int block_x_dim = nr_comps;
        int block_y_dim = std::min((int)sqrt(1024 / nr_comps), 4);  // = 2;
        int block_z_dim = std::min((int)sqrt(1024 / nr_comps), 4);  // = 2;
        dim3 block(block_x_dim, block_y_dim, block_z_dim);

        int grid_x_dim = (nr_comps + block_x_dim - 1) / block_x_dim;
        int grid_y_dim = (output_width + block_y_dim - 1) / block_y_dim;
        int grid_z_dim = (output_height + block_z_dim - 1) / block_z_dim;
        dim3 grid(grid_x_dim, grid_y_dim, grid_z_dim);

        Conv2dModZero<<<grid, block>>>(
            dev_out_label[i], dev_in_label[i], dev_qe_weights,
            dev_qe_bias_label[i], dev_zero_label[modulus], input_width,
            input_height, channel, filter, filter_width, filter_height,
            stride_width, stride_height, nr_comps, modulus);

        cudaCheckError(cudaStreamDestroy(stream[i]));
    }
    cudaDeviceSynchronize();
}

void ocall_init_cuda_aes_engine() { CUDAAESEngine cuda_aes_engine{}; }

void ocall_sign_mult(crt_val_t** dev_out_label, crt_val_t** dev_in_label,
                     crt_val_t** dev_out_label_sign, void* dev_garbler_ciphers,
                     void* dev_evaluator_ciphers, size_t input_size,
                     size_t crt_base_size, size_t crt_base_sum,
                     crt_val_t* crt_base) {
    int crt_base_prefix = 0;
    size_t output_size = input_size;
    size_t nr_blocks = ceil_div(output_size, 32lu);

    cudaStream_t stream[crt_base_size];

    for (size_t i = 0; i < crt_base_size; ++i) {
        crt_val_t q = 2;
        crt_val_t p = crt_base[i];
        size_t nr_comps_q = LabelTensor::get_nr_comps(q);
        size_t nr_comps_p = LabelTensor::get_nr_comps(p);

        cudaCheckError(cudaStreamCreate(&stream[i]));

        eval_sign_mult<<<nr_blocks, 32, 0, stream[i]>>>(
            dev_out_label[i], dev_in_label[i], dev_out_label_sign[0],
            (__uint128_t*)dev_garbler_ciphers,
            (__uint128_t*)dev_evaluator_ciphers, p, q, nr_comps_p, nr_comps_q,
            input_size, crt_base_size, crt_base_sum, crt_base_prefix, i);

        cudaCheckError(cudaStreamDestroy(stream[i]));
        crt_base_prefix += crt_base[i];
    }
    cudaDeviceSynchronize();
}

void ocall_cuda_eval_sign(
    crt_val_t* dev_mrs_label, crt_val_t** dev_in_label, size_t input_size,
    size_t output_size, __uint128_t* dev_approx_res_ciphers,
    mrs_val_t* dev_mrs_base, size_t mrs_base_size, int mrs_base_nr_comps,
    size_t crt_base_size, int crt_base_sum, crt_val_t* crt_base,

    crt_val_t* dev_mrs_sum_most_sig_label, int partial_mrs_base_sum,
    crt_val_t** dev_dev_zero_label, __uint128_t* dev_1_cast_ciphers,
    __uint128_t* dev_2_cast_ciphers,

    crt_val_t** dev_out_label, __uint128_t* dev_sign_ciphers,

    size_t nr_out_moduli, crt_val_t* out_moduli) {
    cudaStream_t stream[crt_base_size];

    size_t nr_blocks = ceil_div(output_size, 32lu);

    int crt_base_prefix = 0;
    for (size_t i = 0; i < crt_base_size; ++i) {
        crt_val_t crt_modulus = crt_base[i];
        cudaCheckError(cudaStreamCreate(&stream[i]));

        eval_approx_res_gadget<<<nr_blocks, 32, 0, stream[i]>>>(
            dev_mrs_label, dev_in_label[i], input_size, dev_approx_res_ciphers,
            dev_mrs_base, mrs_base_size, mrs_base_nr_comps, crt_base_size,
            crt_modulus, crt_base_sum, crt_base_prefix, i);

        crt_base_prefix += crt_base[i];
    }
    cudaDeviceSynchronize();

    // Step 2: Mixed-radix addition
    mrs_sum_most_sig<<<nr_blocks, 32>>>(
        dev_mrs_sum_most_sig_label, dev_mrs_label, input_size, dev_mrs_base,
        mrs_base_size, mrs_base_nr_comps, partial_mrs_base_sum, crt_base_size,
        dev_dev_zero_label, dev_1_cast_ciphers, dev_2_cast_ciphers);

    cudaDeviceSynchronize();

    // Step 3: Check mrs sum for sign
    for (size_t i = 0; i < nr_out_moduli; ++i) {
        crt_val_t out_modulus = out_moduli[i];
        eval_sign<<<nr_blocks, 32, 0, stream[i]>>>(
            dev_out_label[i], dev_mrs_sum_most_sig_label, input_size,
            dev_sign_ciphers, crt_base_size, i, out_modulus, nr_out_moduli,
            dev_mrs_base);
    }

    for (size_t i = 0; i < crt_base_size; ++i) {
        cudaCheckError(cudaStreamDestroy(stream[i]));
    }

    cudaDeviceSynchronize();
}

void ocall_eval_rescale(crt_val_t** dev_out_label, crt_val_t** dev_in_label,
                        crt_val_t** dev_upshift_labels,
                        crt_val_t* dev_zero_label,
                        __uint128_t* dev_trans_mod_ciphers, crt_val_t* crt_base,
                        size_t crt_base_size, size_t input_size) {
    cudaStream_t stream[crt_base_size];

    size_t nr_blocks = ceil_div(input_size, 32lu);

    // Step 1: Upshift by max_crt_modulus / 2
    for (size_t i = 0; i < crt_base_size; ++i) {
        cudaStreamCreate(&stream[i]);

        crt_val_t modulus = crt_base[i];
        size_t nr_comps = LabelTensor::get_nr_comps(modulus);

        AddLabel<<<nr_blocks, 32, 0, stream[i]>>>(
            dev_out_label[i], dev_in_label[i], dev_upshift_labels[i], modulus,
            input_size, nr_comps);
    }
    cudaDeviceSynchronize();

    // Step 2.1
    // a) project residue modulus 2 to modulus crt_base.at(j)
    // b) subtract this value from the input label at
    // modulus crt_base.at(j)...
    // c) multiply by inverse of 2 mod crt_base.at(j)
    for (size_t i = 1; i < crt_base_size; ++i) {
        crt_val_t modulus = crt_base[i];
        crt_val_t inverse_mod = util::mul_inv(2, modulus);
        Scale_1<<<nr_blocks, 32, 0, stream[i]>>>(
            dev_out_label[i], dev_out_label[i], dev_out_label[0],
            dev_trans_mod_ciphers, modulus, input_size, i, crt_base_size,
            inverse_mod);
    }
    cudaDeviceSynchronize();

    // Step 3: Base Extension
    //// Set first result to zero
    size_t nr_components = LabelTensor::get_nr_comps(2);
    DevDevCopy<<<nr_blocks, 32>>>(dev_out_label[0], dev_zero_label, input_size,
                                  nr_components);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < crt_base_size; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}

void ocall_eval_rescale2(crt_val_t** dev_out_label, crt_val_t* dev_sign_out_0,
                         crt_val_t** dev_downshift_labels, size_t input_size,
                         crt_val_t* crt_base, size_t crt_base_size) {
    cudaStream_t stream[crt_base_size];
    size_t nr_blocks = ceil_div(input_size, 32lu);
    size_t nr_components = LabelTensor::get_nr_comps(2);
    DevDevCopy2<<<nr_blocks, 32>>>(dev_out_label[0], dev_sign_out_0, input_size,
                                   nr_components);
    cudaDeviceSynchronize();

    // Step 4: Downshift by max_crt_modulus / 4
    for (size_t i = 0; i < crt_base_size; ++i) {
        cudaStreamCreate(&stream[i]);

        crt_val_t modulus = crt_base[i];
        size_t nr_comps = LabelTensor::get_nr_comps(modulus);

        SubLabel<<<nr_blocks, 32, 0, stream[i]>>>(
            dev_out_label[i], dev_out_label[i], dev_downshift_labels[i],
            modulus, input_size, nr_comps);

        cudaStreamDestroy(stream[i]);
    }
    cudaDeviceSynchronize();
}

/**
 * Main application.
 */
int SGX_CDECL main() {
    init_cuda();
    initialize_enclave();

    std::cout << "Enclave initialized" << std::endl;

    //
    //
    // Dense example
    // ScalarTensor<q_val_t> inputs{{1, 2}, dim_t{2}};
    //
    //

    //
    //
    // Con2D example
    size_t input_width = 3;
    size_t input_height = 3;
    size_t input_size = input_height * input_width;
    size_t channel = 1;
    size_t size = input_size * channel * sizeof(q_val_t);
    q_val_t* plain_inputs = (q_val_t*)malloc(size);

    for (size_t i = 0; i < input_size * channel; ++i) {
        plain_inputs[i] = i;
    }

    dim_t input_dims{input_width, input_height, channel};
    ScalarTensor<q_val_t> inputs{plain_inputs, input_dims};

    free(plain_inputs);
    //
    //
    //

    // ReLU and Sign example
    // ScalarTensor<q_val_t> inputs{{-2, -1, 0, 1, 2}, dim_t{5}};

    // Rescale example
    // vector<q_val_t> in(30);
    // iota(in.begin(), in.end(), -15);
    // ScalarTensor<q_val_t> inputs{in, dim_t{in.size()}};

    //
    //
    // End-to-end example
    // std::cout.precision(17);
    // auto inputs = mnist("../data/MNIST/raw");

    // auto inputs = cifar10("../data/cifar10/cifar-10-batches-bin");
    // vector<wandb_t> mean = {0.4914, 0.4822, 0.4465};
    // vector<wandb_t> std = {0.247, 0.243, 0.261};
    // normalize(&inputs, mean, std);

    //
    //
    //

    sgx_status_t state = ecall_ann_infer(global_eid, &inputs);

    sgx_destroy_enclave(global_eid);

    return 0;
}

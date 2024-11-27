#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda_runtime_api.h>

#include "misc/datatypes.h"

__device__ void cuda_cipher(__uint128_t* out, __uint128_t* in);

// Error checking macro
#define cudaCheckError(code)                                                   \
    {                                                                          \
        if ((code) != cudaSuccess) {                                           \
            fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
                    cudaGetErrorString(code));                                 \
        }                                                                      \
    }

void init_cuda() {
    // expand heap size for dynmic memory allocations in kernels
    size_t size = 32 * 1024 * 1024;  // 32MB
    cudaCheckError(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size));
}

template <typename X>
__device__ __forceinline__ X modulo(X divisor, X dividend) {
    return (divisor % dividend + dividend) % dividend;
}

template <typename V, typename T>
__device__ __forceinline__ T modulo(V divisor, T dividend) {
    V cast_dividend = static_cast<V>(dividend);
    V result = (divisor % cast_dividend + cast_dividend) % cast_dividend;
    return static_cast<T>(result);
}

__device__ __forceinline__ size_t get_nr_comps(crt_val_t modulus) {
    return (size_t)(floor(128 / log2((float)modulus)));
    // >128 bit can not be packed in a single __uint128_t
    // return (size_t)(ceil(128 / log2((float)modulus)));
}

// start just a single thread
__global__ void print_garbled_inputs(crt_val_t* dev_garbled_inputs,
                                     size_t nr_inputs, size_t nr_comps) {
    for (size_t j = 0; j < nr_inputs; ++j) {
        for (size_t k = 0; k < nr_comps; ++k) {
            printf("%d ", dev_garbled_inputs[j * nr_comps + k]);
        }
        printf("\n");
    }
}

__global__ void AddLabel(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                         crt_val_t* dev_in_label2, crt_val_t modulus,
                         size_t nr_inputs, size_t nr_comps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_inputs) {
        for (size_t i = 0; i < nr_comps; ++i) {
            dev_out_label[idx * nr_comps + i] = modulo(
                dev_in_label[idx * nr_comps + i] + dev_in_label2[i], modulus);
        }
    }
}

__global__ void SubLabel(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                         crt_val_t* dev_in_label2, crt_val_t modulus,
                         size_t nr_inputs, size_t nr_comps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_inputs) {
        for (size_t i = 0; i < nr_comps; ++i) {
            dev_out_label[idx * nr_comps + i] = modulo(
                dev_in_label[idx * nr_comps + i] - dev_in_label2[i], modulus);
        }
    }
}

__global__ void MatMulModAdd(const q_val_t* A, const crt_val_t* B, crt_val_t* C,
                             crt_val_t* D, crt_val_t modulus, size_t M,
                             size_t N, size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    crt_val_t tmp = 0;
    crt_val_t sum = 0;
    if ((row < M) && (col < N)) {
        for (size_t i = 0; i < K; ++i) {
            tmp = (A[K * row + i] * B[i * N + col]) % modulus;
            sum = (sum + tmp) % modulus;
        }
        sum = (sum + D[row * N + col]) % modulus;
        C[row * N + col] = sum;
    }
}

__global__ void MatMulModAddZero(const q_val_t* A, const crt_val_t* B,
                                 crt_val_t* C, crt_val_t* D, crt_val_t* Z,
                                 crt_val_t modulus, size_t M, size_t N,
                                 size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    crt_val_t tmp = 0;
    crt_val_t sum = 0;
    if ((row < M) && (col < N)) {
        for (size_t i = 0; i < K; ++i) {
            if (modulo(A[K * row + i], modulus) == 0)
                tmp = Z[col];
            else
                tmp = (A[K * row + i] * B[i * N + col]) % modulus;
            sum = (sum + tmp) % modulus;
        }
        sum = (sum + D[row * N + col]) % modulus;
        C[row * N + col] = sum;
    }
}

__global__ void MatMulModAddZeroTF(const q_val_t* A, const crt_val_t* B,
                                   crt_val_t* C, crt_val_t* D, crt_val_t* Z,
                                   crt_val_t modulus, size_t M, size_t N,
                                   size_t K, size_t channel) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    crt_val_t tmp = 0;
    crt_val_t sum = 0;
    size_t offset = K / channel;
    if ((row < M) && (col < N)) {
        for (size_t i = 0; i < K; ++i) {
            if (modulo(A[K * row + i], modulus) == 0) {
                tmp = Z[col];
            } else {
                size_t input_idx = i / channel + (i % channel) * offset;
                tmp = (A[K * row + i] * B[input_idx * N + col]) % modulus;
            }
            sum = (sum + tmp) % modulus;
        }
        sum = (sum + D[row * N + col]) % modulus;
        C[row * N + col] = sum;
    }
}

__global__ void MatMulModAddTF(const q_val_t* A, const crt_val_t* B,
                               crt_val_t* C, crt_val_t* D, crt_val_t modulus,
                               size_t M, size_t N, size_t K, size_t channel) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    crt_val_t tmp = 0;
    crt_val_t sum = 0;
    size_t offset = K / channel;
    if ((row < M) && (col < N)) {
        for (size_t i = 0; i < K; ++i) {
            size_t input_idx = i / channel + (i % channel) * offset;
            tmp = (A[K * row + i] * B[input_idx * N + col]) % modulus;
            sum = (sum + tmp) % modulus;
        }
        sum = (sum + D[row * N + col]) % modulus;
        C[row * N + col] = sum;
    }
}

__global__ void Conv2dMod(crt_val_t* output, crt_val_t* input, q_val_t* weights,
                          crt_val_t* bias, int input_width, int input_height,
                          int channel, int filter, int filter_width,
                          int filter_height, int stride_width,
                          int stride_height, int nr_comps, crt_val_t modulus) {

    int output_width = (input_width - filter_width) / stride_width + 1;
    int output_height = (input_height - filter_height) / stride_height + 1;

    int output_size = output_width * output_height;  // sizer per filter
    int filter_size = filter_width * filter_height;
    int input_size = input_width * input_height;

    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    int l = threadIdx.z + blockIdx.z * blockDim.z;

    if (m < nr_comps && k < output_width && l < output_height) {
        // for all filter
        for (int v = 0; v < filter; ++v) {
            int output_idx = v * output_size * nr_comps +
                             l * output_width * nr_comps + k * nr_comps + m;
            // scalar product over all input-channel...
            for (int w = 0; w < channel; ++w) {
                // ...along filter height and...
                for (int i = 0; i < filter_height; ++i) {
                    // ...width
                    for (int j = 0; j < filter_width; ++j) {
                        output[output_idx] =
                            (output[output_idx] +
                             weights[v * filter_size * channel +
                                     w * filter_size + i * filter_height + j] *
                                 input[w * input_size * nr_comps +
                                       i * input_width * nr_comps +
                                       j * nr_comps +
                                       k * stride_width * nr_comps +
                                       l * stride_height * input_width *
                                           nr_comps +
                                       m]) %
                            modulus;
                    }
                }
            }
            // add bias to each output-channel
            output[output_idx] =
                (output[output_idx] + bias[v * nr_comps + m]) % modulus;
        }
    }
}

__global__ void Conv2dModZero(crt_val_t* output, crt_val_t* input,
                              q_val_t* weights, crt_val_t* bias,
                              crt_val_t* zero, int input_width,
                              int input_height, int channel, int filter,
                              int filter_width, int filter_height,
                              int stride_width, int stride_height, int nr_comps,
                              crt_val_t modulus) {
    // TODO: Implement cache-tiling?
    int output_width = (input_width - filter_width) / stride_width + 1;
    int output_height = (input_height - filter_height) / stride_height + 1;

    int output_size = output_width * output_height;  // sizer per filter
    int filter_size = filter_width * filter_height;
    int input_size = input_width * input_height;

    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int k = threadIdx.y + blockIdx.y * blockDim.y;
    int l = threadIdx.z + blockIdx.z * blockDim.z;

    if (m < nr_comps && k < output_width && l < output_height) {
        // for all filter
        for (int v = 0; v < filter; ++v) {
            int output_idx = v * output_size * nr_comps +
                             l * output_width * nr_comps + k * nr_comps + m;
            // scalar product over all input-channel...
            for (int w = 0; w < channel; ++w) {
                // ...along filter height and...
                for (int i = 0; i < filter_height; ++i) {
                    // ...width
                    for (int j = 0; j < filter_width; ++j) {
                        if (modulo(weights[v * filter_size * channel +
                                           w * filter_size + i * filter_height +
                                           j],
                                   modulus) == 0) {
                            output[output_idx] =
                                (output[output_idx] + zero[m]) % modulus;
                        } else {
                            output[output_idx] =
                                (output[output_idx] +
                                 weights[v * filter_size * channel +
                                         w * filter_size + i * filter_height +
                                         j] *
                                     input[w * input_size * nr_comps +
                                           i * input_width * nr_comps +
                                           j * nr_comps +
                                           k * stride_width * nr_comps +
                                           l * stride_height * input_width *
                                               nr_comps +
                                           m]) %
                                modulus;
                        }
                    }
                }
            }
            // add bias to each output-channel
            output[output_idx] =
                (output[output_idx] + bias[v * nr_comps + m]) % modulus;
        }
    }
}

__device__ __forceinline__ void compress_label(__uint128_t* compressed,
                                               crt_val_t* label, int nr_comps,
                                               crt_val_t modulus) {
    *compressed = label[nr_comps - 1];
    for (int i = nr_comps - 2; i >= 0; --i) {
        *compressed = *compressed * modulus;
        *compressed += label[i];
    }
}

__device__ __forceinline__ void decompress_label(crt_val_t* label,
                                                 __uint128_t* compressed,
                                                 int nr_comps,
                                                 crt_val_t modulus) {
    for (int i = 0; i < nr_comps; ++i) {
        label[i] = *compressed;
        *compressed /= modulus;
    }
    __uint128_t sub = 0;
    for (int i = nr_comps - 1; i > 0; --i) {
        sub -= label[i] * modulus;
        label[i - 1] += sub;
        sub *= modulus;
    }
    label[nr_comps - 1] %= modulus;
}

__global__ void eval_proj(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                          __uint128_t* ciphers, crt_val_t in_modulus,
                          crt_val_t out_modulus, int nr_comps_in,
                          int nr_comps_out, int nr_label) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nr_label) {
        // generate key
        __uint128_t key_compressed;
        __uint128_t key_hashed;
        compress_label(&key_compressed, &dev_in_label[idx * nr_comps_in],
                       nr_comps_in, in_modulus);
        cuda_cipher(&key_hashed, &key_compressed);

        // decrypt payload
        int color_value = dev_in_label[idx * nr_comps_in];
        __uint128_t p = ciphers[idx * in_modulus + color_value] - key_hashed;
        decompress_label(&dev_out_label[idx * nr_comps_out], &p, nr_comps_out,
                         out_modulus);
    }
}

__device__ void eval_proj(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                          __uint128_t* ciphers, crt_val_t in_modulus,
                          crt_val_t out_modulus, int nr_comps_in,
                          int nr_comps_out) {
    // generate key
    __uint128_t key_compressed;
    __uint128_t key_hashed;
    compress_label(&key_compressed, dev_in_label, nr_comps_in, in_modulus);
    cuda_cipher(&key_hashed, &key_compressed);

    // decrypt payload
    int color_value = dev_in_label[0];
    __uint128_t p = ciphers[color_value] - key_hashed;
    decompress_label(dev_out_label, &p, nr_comps_out, out_modulus);
}

__device__ void eval_mini_proj(crt_val_t* dev_out_label,
                               crt_val_t* dev_in_label, __uint128_t* ciphers,
                               crt_val_t in_modulus, crt_val_t out_modulus,
                               int nr_comps_in) {
    // generate key
    __uint128_t key_compressed;
    __uint128_t key_hashed;
    compress_label(&key_compressed, dev_in_label, nr_comps_in, in_modulus);
    cuda_cipher(&key_hashed, &key_compressed);

    // decrypt payload
    int color_value = dev_in_label[0];
    crt_val_t p = ((crt_val_t*)ciphers)[color_value] - (crt_val_t)key_hashed;
    dev_out_label[0] = p;  // dev_out_label is a single component
}

__host__ __device__ void print_label(crt_val_t* label, int nr_comps,
                                     int nr_label) {
    for (int i = 0; i < nr_label; ++i) {
        for (int j = 0; j < nr_comps; ++j) {
            printf("%d ", label[i * nr_comps + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__device__ void eval_mixed_mod_mult(crt_val_t* out_label,
                                    crt_val_t* dev_garbler_label,
                                    crt_val_t* dev_eval_label,
                                    __uint128_t* dev_garbler_ciphers,
                                    __uint128_t* dev_eval_ciphers, crt_val_t p,
                                    crt_val_t q, int nr_comps_p,
                                    int nr_comps_q) {
    // Decrypt garbler half gate
    auto sk01_xrR = dev_garbler_label;
    crt_val_t garbler_label[128];
    auto cipher = dev_garbler_ciphers;
    eval_proj(garbler_label, sk01_xrR, cipher, p, p, nr_comps_p, nr_comps_p);

    // Decrypt evaluator half gate
    auto sk02_yR = dev_eval_label;
    cipher = dev_eval_ciphers;
    eval_proj(out_label, sk02_yR, cipher, q, p, nr_comps_q, nr_comps_p);

    // Compute result
    // - Decrypt mini projection gate for r+y
    crt_val_t y_plus_r;
    cipher = &dev_eval_ciphers[q];
    eval_mini_proj(&y_plus_r, sk02_yR, cipher, q, p, nr_comps_q);

    // - Compute output label
    for (int i = 0; i < nr_comps_p; ++i) {
        crt_val_t tmp = modulo(y_plus_r * sk01_xrR[i], p);
        out_label[i] = modulo(out_label[i] + tmp, p);
        out_label[i] = modulo(out_label[i] - garbler_label[i], p);
    }
}

__global__ void DevDevCopy(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                           size_t nr_inputs, size_t nr_comps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_inputs) {
        memcpy(dev_out_label + idx * nr_comps, dev_in_label,
               nr_comps * sizeof(crt_val_t));
    }
}

__global__ void DevDevCopy2(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                            size_t nr_inputs, size_t nr_comps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_inputs) {
        memcpy(dev_out_label + idx * nr_comps, dev_in_label + idx * nr_comps,
               nr_comps * sizeof(crt_val_t));
    }
}

__global__ void Scale_1(crt_val_t* dev_out_label, crt_val_t* dev_in_label,
                        crt_val_t* dev_crt0_labels,
                        __uint128_t* dev_trans_mod_ciphers, crt_val_t modulus,
                        size_t nr_inputs, size_t crt_idx, size_t crt_base_size,
                        crt_val_t inverse_mod) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_inputs) {
        // Step 2.1.a
        size_t in_modulus = 2;
        size_t in_nr_comps = get_nr_comps(in_modulus);
        size_t out_nr_comps = get_nr_comps(modulus);
        crt_val_t mod_2_label[128];
        crt_val_t* crt0_label = dev_crt0_labels + idx * in_nr_comps;

        __uint128_t* c = dev_trans_mod_ciphers;
        c += idx * (crt_base_size - 1) * 2;
        c += (crt_idx - 1) * 2;

        eval_proj(mod_2_label, crt0_label, c, in_modulus, modulus, in_nr_comps,
                  out_nr_comps);

        // Step 2.1.b
        for (size_t i = 0; i < out_nr_comps; ++i) {
            dev_out_label[idx * out_nr_comps + i] = modulo(
                dev_in_label[idx * out_nr_comps + i] - mod_2_label[i], modulus);
        }

        // Step 2.1.c
        for (size_t i = 0; i < out_nr_comps; ++i) {
            dev_out_label[idx * out_nr_comps + i] = modulo(
                dev_out_label[idx * out_nr_comps + i] * inverse_mod, modulus);
        }
    }
}
#endif
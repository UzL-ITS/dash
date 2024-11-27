#ifndef TEST_CUDA_AES_ENGINE_H
#define TEST_CUDA_AES_ENGINE_H

#include "crypto/cuda_aes_engine.h"
#include "misc/cuda_util.h"
#include "misc/datatypes.h"
#include "garbling/label_tensor.h"

using std::vector;

class TestCudaAesEngine : public ::testing::Test {};

__host__ __device__ void print_hex(unsigned char* data, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

__global__ void test_hash(__uint128_t* ciphertext, __uint128_t* plaintext,
                          int nr_label) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_label) cuda_cipher(ciphertext + idx, plaintext + idx);
}

TEST(TestCudaAesEngine, CudaHash) {
    // Initialize CUDAAESEngine
    CUDAAESEngine engine{};

    size_t nr_label = 100;
    LabelTensor test{19, dim_t{nr_label}};
    test.init_random();
    test.compress();
    test.hash();
    __uint128_t* compressed_dev;
    __uint128_t test_hashed[100];

    cudaCheckError(
        cudaMalloc((void**)&compressed_dev, nr_label * sizeof(__uint128_t)));
    cudaCheckError(cudaMemcpy(compressed_dev, test.get_compressed(),
                              nr_label * sizeof(__uint128_t),
                              cudaMemcpyHostToDevice));

    __uint128_t* hashed_dev;
    cudaCheckError(cudaMalloc((void**)&hashed_dev, nr_label* sizeof(__uint128_t)));

    test_hash<<<4, 32>>>(hashed_dev, compressed_dev, nr_label);

    cudaCheckError(cudaMemcpy(&test_hashed, hashed_dev, nr_label * sizeof(__uint128_t),
                              cudaMemcpyDeviceToHost));

    for (int i = 0; i < nr_label; i++) {
        EXPECT_EQ(test.get_hashed()[i], test_hashed[i]);
    }

    cudaCheckError(cudaFree(compressed_dev));
    cudaCheckError(cudaFree(hashed_dev));
}

#endif
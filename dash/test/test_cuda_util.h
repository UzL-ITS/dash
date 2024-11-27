#ifndef TEST_CUDA_UTIL_H
#define TEST_CUDA_UTIL_H

#include "misc/cuda_util.h"
#include "garbling/label_tensor.h"

class TestCudaUtil : public ::testing::Test {};

__global__ void test_compress(__uint128_t* compressed, crt_val_t* label,
                              int nr_label, int nr_comps, crt_val_t modulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_label)
        compress_label(compressed + idx, label + idx * nr_comps, nr_comps,
                       modulus);
}

__global__ void test_decompress(crt_val_t* label, __uint128_t* compressed,
                                int nr_label, int nr_comps, crt_val_t modulus) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr_label)
        decompress_label(label + idx * nr_comps, compressed + idx, nr_comps,
                         modulus);
}

TEST(TestCudaUtil, TestCudaCompressLabel) {
    size_t nr_label = 100;
    LabelTensor test{19, dim_t{nr_label}};
    test.init_random();
    test.compress();

    crt_val_t* label_dev;
    cudaCheckError(
        cudaMalloc((void**)&label_dev, test.size() * sizeof(crt_val_t)));
    cudaCheckError(cudaMemcpy(label_dev, test.get_components(),
                              test.size() * sizeof(crt_val_t),
                              cudaMemcpyHostToDevice));

    __uint128_t* compressed_dev;
    cudaCheckError(
        cudaMalloc((void**)&compressed_dev, nr_label * sizeof(__uint128_t)));

    test_compress<<<4, 32>>>(compressed_dev, label_dev, test.get_nr_label(),
                             test.get_nr_comps(), test.get_modulus());

    auto compressed_host = new __uint128_t[nr_label];

    cudaCheckError(cudaMemcpy(compressed_host, compressed_dev,
                              nr_label * sizeof(__uint128_t),
                              cudaMemcpyDeviceToHost));

    for (int i = 0; i < nr_label; i++) {
        EXPECT_EQ(test.get_compressed()[i], compressed_host[i]);
    }

    delete[] compressed_host;
    cudaCheckError(cudaFree(compressed_dev));
    cudaCheckError(cudaFree(label_dev));
}

TEST(TestCudaUtil, TestCudaDecompressLabel) {
    size_t nr_label = 100;
    LabelTensor test{19, dim_t{nr_label}};
    test.init_random();
    test.compress();

    crt_val_t* label_dev;
    cudaCheckError(
        cudaMalloc((void**)&label_dev, test.size() * sizeof(crt_val_t)));
    cudaCheckError(cudaMemcpy(label_dev, test.get_components(),
                              test.size() * sizeof(crt_val_t),
                              cudaMemcpyHostToDevice));

    crt_val_t* label_decompress_dev;
    cudaCheckError(cudaMalloc((void**)&label_decompress_dev,
                              test.size() * sizeof(crt_val_t)));

    __uint128_t* compressed_dev;
    cudaCheckError(
        cudaMalloc((void**)&compressed_dev, nr_label * sizeof(__uint128_t)));

    test_compress<<<4, 32>>>(compressed_dev, label_dev, test.get_nr_label(),
                             test.get_nr_comps(), test.get_modulus());

    test_decompress<<<4, 32>>>(label_decompress_dev, compressed_dev,
                               test.get_nr_label(), test.get_nr_comps(),
                               test.get_modulus());

    auto compressed_host = new __uint128_t[nr_label];

    cudaCheckError(cudaMemcpy(compressed_host, compressed_dev,
                              nr_label * sizeof(__uint128_t),
                              cudaMemcpyDeviceToHost));

    auto decompressed_host = new crt_val_t[nr_label * test.get_nr_comps()];
    cudaCheckError(
        cudaMemcpy(decompressed_host, label_decompress_dev,
                   nr_label * test.get_nr_comps() * sizeof(crt_val_t),
                   cudaMemcpyDeviceToHost));

    vector<crt_val_t> decompressed(
        decompressed_host, decompressed_host + nr_label * test.get_nr_comps());
    EXPECT_EQ(test.as_vector(), decompressed);
    std::cout << std::endl;

    delete[] compressed_host;
    delete[] decompressed_host;
    cudaCheckError(cudaFree(compressed_dev));
    cudaCheckError(cudaFree(label_dev));
    cudaCheckError(cudaFree(label_decompress_dev));
}

#endif
# Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits

This repository contains the proof-of-concept implementation of the Dash framework. Do not use this software in production environments.

Dash is a framework for accelerated distributed private convolutional neural network inference and provides security against a *malicious adversary*. Building on arithmetic garbling gadgets [1] and fancy-garbling [2], Dash is based purely on *arithmetic garbled circuits*. Dash uses *LabelTensors* that allow to leverage the massive parallelity of modern GPUs. Combined with state-of-the-art garbling optimizations, Dash outperforms previous garbling approaches up to a factor of about 100. Furthermore, Dash provides an efficient *scaling* operation over the residues of the Chinese remainder theorem representation to arithmetic garbled circuits, which allows to garble larger networks and achieve much higher accuracy than previous approaches. Finally, Dash requires only a single communication round per inference step, regardless of the depth of the neural network, and a very small constant online communication volume.

## Organization
You can find the benchmarks of our paper in the `benchmarks` folder. The code to train the models used in our paper is in the `models` folder. A simple example of how to use Dash is in the `dash/example` folder or if you want to use SGX in the `sgx` folder. In the follwing, we describe the setup and requirements to run Dash. 

## Setup and Dependencies
Dash was developed, benchmarked, and tested on the following system:
- Ubuntu Server 20.04.5 LTS (Kernel 5.11.0-051100-generic)
- Intel(R) Xeon(R) Gold 5415+
- Nvidia RTX 4090

With the following software:
- GCC version 9.4.0
- Intel SGX Linux 2.18 (with in-kernel driver)
- Google's Protocol Buffers for C++ v3.20
- dlib
- CUDA 12.2
- gtest

We used Python 3.9.10 with the dependencies from the `requirements.txt` installed to train the models and run the notebooks.

## Building and Running Dash
1. Copy the files from the following directory to the enclave includes (be aware, we used the gcc version 9.4.0 and your path may differ):
```bash
mkdir -p sgx/Enclave/include/intrinsics
cp -r /usr/lib/gcc/x86_64-linux-gnu/9/include/* sgx/Enclave/include/intrinsics

mkdir -p benchmarks/model_benchmarks/sgx/Enclave/include/intrinsics
cp -r /usr/lib/gcc/x86_64-linux-gnu/9/include/* benchmarks/model_benchmarks/sgx/Enclave/include/intrinsics
```
2. Download onnx.proto3 from the onnx repository:
```bash
wget -P dash/onnx https://raw.githubusercontent.com/onnx/onnx/093a8d335a66ea136eb1f16b3a1ce6237ee353ab/onnx/onnx.proto3
```
3. Generate private keys for the enclave:
```bash
openssl genrsa -3 3072 > sgx/Enclave/private_key.pem

openssl genrsa -3 3072 > benchmarks/model_benchmarks/sgx/Enclave/private_key.pem
```
4. To download the datasets run `cd data` and `.download.sh`.
5. To train the models from the paper or visualize the benchmark results run the corresponding notebooks in the `models` and `benchmarks` folder.
6. To run the example or the benchmarks change to the corresponding directory, run the Makefile `make release` and execute the binary `./main`. If you want to run an example with SGX, use `make SGX_PRERELEASE=1 SGX_DEBUG=0`to build the SGX enclave and the binary. The results can be visusalized with evaluation.ipynb in the `benchmarks` folder.

## Paper
For a detailed description of the framework, please refer to our paper:

Jonas Sander, Sebastian Berndt, Ida Bruhns and Thomas Eisenbarth. 2025. **Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits.** IACR Transactions on Cryptographic Hardware and Embedded Systems, 2025(1), 420-449. \[[Link](https://tches.iacr.org/index.php/TCHES/article/view/11935)\] \[[DOI](https://doi.org/10.46586/tches.v2025.i1.420-449)\]

## References
[1] Ball, Marshall, Tal Malkin, and Mike Rosulek. "Garbling gadgets for boolean and arithmetic circuits." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. 2016.

[2] Ball, Marshall, et al. "Garbled neural networks are practical." Cryptology ePrint Archive (2019).

## Disclaimer
The offline phase is not optimized at the moment.

## License
The SGX code is based on the Intel SGX Example Enclave which is licensed under BSD license. The rest of the code is licensed under the GPL-3.0 License.
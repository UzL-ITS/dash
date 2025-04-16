#include <chrono>
#include <random>
#include <vector>

#include "../../benchmark_utilities.h"
#include "circuit/circuit.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/sign.h"
#include "circuit/scalar_tensor.h"
#include "garbling/garbled_circuit.h"
#include "misc/datatypes.h"

using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Initialize weights like pytorch
ScalarTensor<wandb_t> init_wb(dim_t dims, std::mt19937& gen) {
    double in_features = dims.at(0);
    double bound = std::sqrt(1.0 / in_features);
    // Sample from uniform from [-bound, bound)
    std::uniform_real_distribution<> dis(-bound, bound);
    ScalarTensor<wandb_t> weights{dims};
    for (size_t i = 0; i < weights.get_capacity(); i++) {
        weights.push_back(dis(gen));
    }
    return weights;
}

ScalarTensor<wandb_t> init_w_conv2d(dim_t dims, std::mt19937& gen) {
    double groups = 1;
    double filter_size = dims.at(0) * dims.at(1);
    double channel = dims.at(2);
    double bound = std::sqrt(groups / (channel * filter_size));
    std::uniform_real_distribution<> dis(-bound, bound);
    ScalarTensor<wandb_t> weights{dims};
    for (size_t i = 0; i < weights.get_capacity(); i++) {
        weights.push_back(dis(gen));
    }
    return weights;
}

ScalarTensor<wandb_t> init_b_conv2d(dim_t dims, int filter_size, int channel,
                                    std::mt19937& gen) {
    double groups = 1;
    double bound = std::sqrt(groups / (channel * filter_size));
    std::uniform_real_distribution<> dis(-bound, bound);
    ScalarTensor<wandb_t> weights{dims};
    for (size_t i = 0; i < weights.get_capacity(); i++) {
        weights.push_back(dis(gen));
    }
    return weights;
}

// Sample inputs unifrom from [0,255]
ScalarTensor<q_val_t> init_inputs(dim_t dims, std::mt19937& gen) {
    std::uniform_int_distribution<q_val_t> dis(0, 255);
    ScalarTensor<q_val_t> inputs{dims};
    for (size_t i = 0; i < inputs.get_capacity(); i++) {
        inputs.push_back(dis(gen));
    }
    return inputs;
}

double get_mem_usage() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    return used_db;
}

void bench_dense_cpu(std::mt19937 gen, int runs, dim_t dims, wandb_t q_const,
                     FILE* fpt) {
    for (auto dim : dims) {
        printf("Dense Layer (CPU), dim: %lu\n", dim);
        for (int run = 0; run < runs; ++run) {
            size_t in_features = dim;
            size_t out_features = dim;
            auto weights = init_wb(dim_t{in_features, out_features}, gen);
            auto biases = init_wb(dim_t{out_features}, gen);
            auto inputs_q = init_inputs(dim_t{in_features}, gen);
            auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                inputs_q.data(), inputs_q.get_dims());

            auto circuit = new Circuit{new Dense{
                weights, biases, -1, QuantizationMethod::SimpleQuant, q_const}};
            auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
            int crt_base_size = circuit->infer_crt_base_size(inputs_q);
            auto gc = new GarbledCircuit(circuit, crt_base_size);
            auto g_inputs{gc->garble_inputs(inputs_q)};

            auto t1 = high_resolution_clock::now();
            auto g_outputs{gc->cpu_evaluate(g_inputs)};
            auto t2 = high_resolution_clock::now();
            auto outputs{gc->decode_outputs(g_outputs)};

            duration<double, std::milli> ms_double = t2 - t1;

            fprintf(fpt, "CPU, %lu, %d, %d, %f, %f, %f\n", dim, crt_base_size,
                    run, ms_double.count(), -1.0, q_acc);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;
            delete circuit;
            delete gc;
        }
    }
}

void bench_dense_gpu(std::mt19937 gen, int runs, dim_t dims, wandb_t q_const,
                     FILE* fpt) {
    for (auto dim : dims) {
        printf("Dense Layer (GPU), dim: %lu\n", dim);
        for (int run = 0; run < runs; ++run) {
            auto m1 = get_mem_usage();

            size_t in_features = dim;
            size_t out_features = dim;
            auto weights = init_wb(dim_t{in_features, out_features}, gen);
            auto biases = init_wb(dim_t{out_features}, gen);
            auto inputs_q = init_inputs(dim_t{in_features}, gen);
            auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                inputs_q.data(), inputs_q.get_dims());

            auto circuit = new Circuit{new Dense{
                weights, biases, -1, QuantizationMethod::SimpleQuant, q_const}};
            auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
            int crt_base_size = circuit->infer_crt_base_size(inputs_q);
            auto gc = new GarbledCircuit(circuit, crt_base_size);
            gc->cuda_move();
            auto g_inputs{gc->garble_inputs(inputs_q)};
            auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
            auto m2 = get_mem_usage();

            auto mem_usage = m2 - m1;

            auto t1 = high_resolution_clock::now();
            gc->cuda_evaluate(g_dev_inputs);
            auto t2 = high_resolution_clock::now();
            auto g_outputs{gc->cuda_move_outputs()};
            auto outputs{gc->decode_outputs(g_outputs)};

            duration<double, std::milli> ms_double = t2 - t1;

            fprintf(fpt, "GPU, %lu, %d, %d, %f, %f, %f\n", dim, crt_base_size,
                    run, ms_double.count(), mem_usage, q_acc);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;
            gc->cuda_free_inputs(g_dev_inputs);
            delete circuit;
            delete gc;
        }
    }
}

void bench_conv2d_cpu(std::mt19937 gen, int runs, vector<dim_t> dims_vec,
                      wandb_t q_const, FILE* fpt) {
    for (auto dims : dims_vec) {
        printf("Conv2D Layer (CPU), dims: ");
        for (auto dim : dims) {
            printf("%lu ", dim);
        }
        printf("\n");

        for (int run = 0; run < runs; ++run) {
            size_t input_width = dims.at(0);
            size_t input_height = dims.at(1);
            size_t channel = dims.at(2);
            size_t input_size = input_height * input_width * channel;

            size_t filter = 16;
            size_t filter_width = 4;
            size_t filter_height = 4;
            size_t filter_size = filter_width * filter_height;

            size_t stride_width = 2;
            size_t stride_height = 2;

            dim_t weights_dims{filter_width, filter_height, channel, filter};
            auto weights = init_w_conv2d(weights_dims, gen);
            dim_t bias_dims{filter};
            auto bias = init_b_conv2d(bias_dims, filter_size, channel, gen);
            dim_t input_dims{input_width, input_height, channel};
            auto inputs_q = init_inputs(input_dims, gen);
            auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                inputs_q.data(), inputs_q.get_dims());

            auto circuit = new Circuit{new Conv2d(
                weights, bias, input_width, input_height, channel, filter,
                filter_width, filter_height, stride_width, stride_height,
                -1, QuantizationMethod::SimpleQuant, q_const)};
            auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
            int crt_base_size = circuit->infer_crt_base_size(inputs_q);
            auto gc = new GarbledCircuit(circuit, crt_base_size);
            auto g_inputs{gc->garble_inputs(inputs_q)};

            auto t1 = high_resolution_clock::now();
            auto g_outputs{gc->cpu_evaluate(g_inputs)};
            auto t2 = high_resolution_clock::now();
            auto outputs{gc->decode_outputs(g_outputs)};

            duration<double, std::milli> ms_double = t2 - t1;

            fprintf(fpt, "CPU, %lu, %d, %d, %f, %f, %f\n", input_size,
                    crt_base_size, run, ms_double.count(), -1.0, q_acc);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;

            delete circuit;
            delete gc;
        }
    }
}

void bench_conv2d_gpu(std::mt19937 gen, int runs, vector<dim_t> dims_vec,
                      wandb_t q_const, FILE* fpt) {
    for (auto dims : dims_vec) {
        printf("Conv2D Layer (GPU), dims: ");
        for (auto dim : dims) {
            printf("%lu ", dim);
        }
        printf("\n");

        for (int run = 0; run < runs; ++run) {
            auto m1 = get_mem_usage();
            size_t input_width = dims.at(0);
            size_t input_height = dims.at(1);
            size_t channel = dims.at(2);
            size_t input_size = input_height * input_width * channel;

            size_t filter = 16;
            size_t filter_width = 4;
            size_t filter_height = 4;
            size_t filter_size = filter_width * filter_height;

            size_t stride_width = 2;
            size_t stride_height = 2;

            dim_t weights_dims{filter_width, filter_height, channel, filter};
            auto weights = init_w_conv2d(weights_dims, gen);
            dim_t bias_dims{filter};
            auto bias = init_b_conv2d(bias_dims, filter_size, channel, gen);
            dim_t input_dims{input_width, input_height, channel};
            auto inputs_q = init_inputs(input_dims, gen);
            auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                inputs_q.data(), inputs_q.get_dims());

            auto circuit = new Circuit{new Conv2d(
                weights, bias, input_width, input_height, channel, filter,
                filter_width, filter_height, stride_width, stride_height,
                5, QuantizationMethod::SimpleQuant, q_const)};
            auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
            int crt_base_size = circuit->infer_crt_base_size(inputs_q);
            auto gc = new GarbledCircuit(circuit, crt_base_size);
            gc->cuda_move();
            auto g_inputs{gc->garble_inputs(inputs_q)};
            auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
            auto m2 = get_mem_usage();

            auto mem_usage = m2 - m1;

            auto t1 = high_resolution_clock::now();
            gc->cuda_evaluate(g_dev_inputs);
            auto t2 = high_resolution_clock::now();
            auto g_outputs{gc->cuda_move_outputs()};
            auto outputs{gc->decode_outputs(g_outputs)};

            duration<double, std::milli> ms_double = t2 - t1;

            fprintf(fpt, "GPU, %lu, %d, %d, %f, %f, %f\n", input_size,
                    crt_base_size, run, ms_double.count(), mem_usage, q_acc);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;
            gc->cuda_free_inputs(g_dev_inputs);

            delete circuit;
            delete gc;
        }
    }
}

void bench_approx_relu_cpu(std::mt19937 gen, int runs, dim_t dims,
                           wandb_t q_const, FILE* fpt) {
    vector<float> relu_accs{99.0, 100.0};
    for (auto dim : dims) {
        printf("Approx. ReLU Layer (CPU), dim: %lu\n", dim);
        for (auto relu_acc : relu_accs) {
            for (int run = 0; run < runs; ++run) {
                auto inputs_q = init_inputs(dim_t{dim}, gen);
                auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                    inputs_q.data(), inputs_q.get_dims());

                auto circuit = new Circuit{new Relu{inputs.get_dims()}};
                auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
                // int crt_base_size = circuit->infer_crt_base_size(inputs);
                int crt_base_size = 8;

                auto gc = new GarbledCircuit(circuit, crt_base_size, relu_acc);
                auto g_inputs{gc->garble_inputs(inputs_q)};
                auto t1 = high_resolution_clock::now();
                auto g_outputs{gc->cpu_evaluate(g_inputs)};
                auto t2 = high_resolution_clock::now();
                auto outputs{gc->decode_outputs(g_outputs)};

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "CPU, %lu, %d, %d, %f, %f, %f, %f\n", dim,
                        crt_base_size, run, ms_double.count(), -1.0, q_acc,
                        relu_acc);

                // clean up
                for (auto label : *g_inputs) {
                    delete label;
                }
                delete g_inputs;

                delete circuit;
                delete gc;
            }
        }
    }
}

void bench_approx_relu_gpu(std::mt19937 gen, int runs, dim_t dims,
                           wandb_t q_const, FILE* fpt) {
    vector<float> relu_accs{99.0, 100.0};
    for (auto dim : dims) {
        printf("Approx. ReLU Layer (GPU), dim: %lu\n", dim);
        for (auto relu_acc : relu_accs) {
            for (int run = 0; run < runs; ++run) {
                auto inputs_q = init_inputs(dim_t{dim}, gen);
                auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                    inputs_q.data(), inputs_q.get_dims());

                auto circuit = new Circuit{new Relu{inputs.get_dims()}};
                auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
                // int crt_base_size = circuit->infer_crt_base_size(inputs);
                int crt_base_size = 8;

                auto m1 = get_mem_usage();
                auto gc = new GarbledCircuit(circuit, crt_base_size, relu_acc);
                gc->cuda_move();
                auto g_inputs{gc->garble_inputs(inputs_q)};
                auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
                auto m2 = get_mem_usage();

                auto mem_usage = m2 - m1;

                auto t1 = high_resolution_clock::now();
                gc->cuda_evaluate(g_dev_inputs);
                auto t2 = high_resolution_clock::now();
                auto g_outputs{gc->cuda_move_outputs()};
                auto outputs{gc->decode_outputs(g_outputs)};

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "GPU, %lu, %d, %d, %f, %f, %f, %f\n", dim,
                        crt_base_size, run, ms_double.count(), mem_usage, q_acc,
                        relu_acc);

                // clean up
                for (auto label : *g_inputs) {
                    delete label;
                }
                delete g_inputs;
                gc->cuda_free_inputs(g_dev_inputs);

                delete circuit;
                delete gc;
            }
        }
    }
}

void bench_sign_activation_cpu(std::mt19937 gen, int runs, dim_t dims,
                               wandb_t q_const, FILE* fpt) {
    for (auto dim : dims) {
        vector<float> sign_accs{99.0, 100.0};
        printf("Sign Activation Layer (CPU), dim: %lu\n", dim);
        for (auto sign_acc : sign_accs) {
            for (int run = 0; run < runs; ++run) {
                auto inputs_q = init_inputs(dim_t{dim}, gen);
                auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                    inputs_q.data(), inputs_q.get_dims());

                auto circuit = new Circuit{new Sign{inputs.get_dims()}};
                auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
                // int crt_base_size = circuit->infer_crt_base_size(inputs);
                int crt_base_size = 8;

                auto gc = new GarbledCircuit(circuit, crt_base_size, sign_acc);
                auto g_inputs{gc->garble_inputs(inputs_q)};
                auto t1 = high_resolution_clock::now();
                auto g_outputs{gc->cpu_evaluate(g_inputs)};
                auto t2 = high_resolution_clock::now();
                auto outputs{gc->decode_outputs(g_outputs)};

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "CPU, %lu, %d, %d, %f, %f, %f, %f\n", dim,
                        crt_base_size, run, ms_double.count(), -1.0, q_acc,
                        sign_acc);

                // clean up
                for (auto label : *g_inputs) {
                    delete label;
                }
                delete g_inputs;

                delete circuit;
                delete gc;
            }
        }
    }
}

void bench_sign_activation_gpu(std::mt19937 gen, int runs, dim_t dims,
                               wandb_t q_const, FILE* fpt) {
    for (auto dim : dims) {
        vector<float> sign_accs{99.0, 100.0};
        printf("Sign Activation Layer (GPU), dim: %lu\n", dim);
        for (auto sign_acc : sign_accs) {
            for (int run = 0; run < runs; ++run) {
                auto inputs_q = init_inputs(dim_t{dim}, gen);
                auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                    inputs_q.data(), inputs_q.get_dims());

                auto circuit = new Circuit{new Sign{inputs.get_dims()}};
                auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
                // int crt_base_size = circuit->infer_crt_base_size(inputs);
                int crt_base_size = 8;

                auto m1 = get_mem_usage();
                auto gc = new GarbledCircuit(circuit, crt_base_size, sign_acc);
                gc->cuda_move();
                auto g_inputs{gc->garble_inputs(inputs_q)};
                auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
                auto m2 = get_mem_usage();

                auto mem_usage = m2 - m1;

                auto t1 = high_resolution_clock::now();
                gc->cuda_evaluate(g_dev_inputs);
                auto t2 = high_resolution_clock::now();
                auto g_outputs{gc->cuda_move_outputs()};
                auto outputs{gc->decode_outputs(g_outputs)};

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "GPU, %lu, %d, %d, %f, %f, %f, %f\n", dim,
                        crt_base_size, run, ms_double.count(), mem_usage, q_acc,
                        sign_acc);

                // clean up
                for (auto label : *g_inputs) {
                    delete label;
                }
                delete g_inputs;
                gc->cuda_free_inputs(g_dev_inputs);

                delete circuit;
                delete gc;
            }
        }
    }
}

void bench_rescaling_cpu(std::mt19937 gen, int runs, dim_t dims,
                         wandb_t q_const, FILE* fpt, bool use_legacy_scaling) {
    for (auto dim : dims) {
        if(use_legacy_scaling) {
            printf("Rescaling Layer (CPU) (DASH's old scaling), dim: %lu\n", dim);
        } else {
            printf("Rescaling Layer (CPU) (ReDASH's new scaling), dim: %lu\n", dim);
        }
        
        for (int run = 0; run < runs; ++run) {
            auto inputs_q = init_inputs(dim_t{dim}, gen);

            Circuit* circuit;
            if (use_legacy_scaling) {
                circuit = new Circuit{new Rescale{1, inputs_q.get_dims()}};
            } else {
                const vector<crt_val_t> s = {2};
                circuit = new Circuit{new Rescale{s, inputs_q.get_dims()}};
            }

            int crt_base_size = 8;

            auto gc = new GarbledCircuit(circuit, crt_base_size, 100.0F);
            auto g_inputs{gc->garble_inputs(inputs_q)};
            auto t1 = high_resolution_clock::now();
            auto g_outputs{gc->cpu_evaluate(g_inputs)};
            auto t2 = high_resolution_clock::now();
            auto outputs{gc->decode_outputs(g_outputs)};

            duration<double, std::milli> ms_double = t2 - t1;

            fprintf(fpt, "CPU, %lu, %d, %d, %f, %f, %d\n", dim, crt_base_size, run,
                    ms_double.count(), -1.0, use_legacy_scaling);

            // clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;

            delete circuit;
            delete gc;
        }
    }
}

// TODO: port to redash
// void bench_rescaling_gpu(std::mt19937 gen, int runs, dim_t dims,
//                          wandb_t q_const, FILE* fpt) {
//     for (auto dim : dims) {
//         printf("Rescaling Layer (CPU), dim: %lu\n", dim);
//         for (int run = 0; run < runs; ++run) {
//             auto inputs_q = init_inputs(dim_t{dim}, gen);

//             auto circuit = new Circuit{new Rescale{1, inputs_q.get_dims()}};
//             int crt_base_size = 8;

//             auto m1 = get_mem_usage();
//             auto gc = new GarbledCircuit(circuit, crt_base_size, 100.0F);
//             gc->cuda_move();
//             auto g_inputs{gc->garble_inputs(inputs_q)};
//             auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//             auto m2 = get_mem_usage();

//             auto mem_usage = m2 - m1;

//             auto t1 = high_resolution_clock::now();
//             gc->cuda_evaluate(g_dev_inputs);
//             auto t2 = high_resolution_clock::now();
//             auto g_outputs{gc->cuda_move_outputs()};
//             auto outputs{gc->decode_outputs(g_outputs)};

//             duration<double, std::milli> ms_double = t2 - t1;

//             fprintf(fpt, "GPU, %lu, %d, %d, %f, %f\n", dim, crt_base_size, run,
//                     ms_double.count(), mem_usage);

//             // clean up
//             for (auto label : *g_inputs) {
//                 delete label;
//             }
//             delete g_inputs;

//             delete circuit;
//             delete gc;
//         }
//     }
// }

int main() {
    init_cuda();
    std::mt19937 gen(42);
    FILE* fpt;
    std::string path = "../data/";
    auto date_string = get_date_string();

    create_dir(path);

    int runs = 10;
    //
    //
    // Benchmark: Dense Layer
    {
        std::string filename = path + date_string + "_dense.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt,
                "type, dimensions, crt_base_size, run, runtime, gpu_mem_usage, "
                "q_acc\n");

        wandb_t q_const = 0.0001;
        dim_t dims = {128, 256, 512, 1024, 2048};
        // bench_dense_cpu(gen, runs, dims, q_const, fpt);
        // bench_dense_gpu(gen, runs, dims, q_const, fpt);

        fclose(fpt);
    }
    //
    //
    //

    //
    //
    // Benchmark: Conv2D Layer
    {
        std::string filename = path + date_string + "_conv2d.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt,
                "type, dimensions, crt_base_size, run, runtime, gpu_mem_usage, "
                "q_acc\n");

        wandb_t q_const = 0.0001;
        vector<dim_t> dims_vec = {{64, 64, 3}, {128, 128, 3}, {256, 256, 3}};
        // bench_conv2d_cpu(gen, runs, dims_vec, q_const, fpt);
        // bench_conv2d_gpu(gen, runs, dims_vec, q_const, fpt);
        fclose(fpt);
    }
    //
    //
    //

    //
    //
    // Benchmark: Approx. ReLu
    {
        std::string filename = path + date_string + "_approx_relu.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt,
                "type, dimensions, crt_base_size, run, runtime, gpu_mem_usage, "
                "q_acc, relu_acc\n");
        wandb_t q_const = 0.0001;
        dim_t dims = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        // bench_approx_relu_cpu(gen, runs, dims, q_const, fpt);
        // bench_approx_relu_gpu(gen, runs, dims, q_const, fpt);
        fclose(fpt);
    }
    //
    //
    //

    //
    //
    // Benchmark: Sign Activation
    {
        std::string filename = path + date_string + "_sign_activation.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt,
                "type, dimensions, crt_base_size, run, runtime, gpu_mem_usage, "
                "q_acc, sign_acc\n");
        wandb_t q_const = 0.0001;
        dim_t dims = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        // bench_sign_activation_cpu(gen, runs, dims, q_const, fpt);
        // bench_sign_activation_gpu(gen, runs, dims, q_const, fpt);
        fclose(fpt);
    }
    //
    //
    //

    //
    //
    // Benchmark: Rescaling
    {
        std::string filename = path + date_string + "_rescaling.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(
            fpt,
            "type, dimensions, crt_base_size, run, runtime, gpu_mem_usage, use_legacy_scaling\n");
        wandb_t q_const = 0.0001;
        dim_t dims = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
        bench_rescaling_cpu(gen, runs, dims, q_const, fpt, true);
        bench_rescaling_cpu(gen, runs, dims, q_const, fpt, false);
        // bench_rescaling_gpu(gen, runs, dims, q_const, fpt); // TODO: port to redash
        fclose(fpt);
    }
    //
    //
    //
}

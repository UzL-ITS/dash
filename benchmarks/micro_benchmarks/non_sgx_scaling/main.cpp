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
ScalarTensor<wandb_t> init_wb(dim_t dims) {
    std::mt19937 gen(42);
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

ScalarTensor<wandb_t> init_w_conv2d(dim_t dims) {
    std::mt19937 gen(42);
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

ScalarTensor<wandb_t> init_b_conv2d(dim_t dims, int filter_size, int channel) {
    std::mt19937 gen(42);
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
ScalarTensor<q_val_t> init_inputs(dim_t dims) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<q_val_t> dis(0, 255);
    ScalarTensor<q_val_t> inputs{dims};
    for (size_t i = 0; i < inputs.get_capacity(); i++) {
        inputs.push_back(dis(gen));
    }
    return inputs;
}

void bench_dense_cpu(int runs, wandb_t q_const,
                     vector<size_t> dims, FILE* fpt) {
    for (auto dim : dims) {
        for (int nr_threads = 1; nr_threads <= 16; ++nr_threads) {
            printf("Dense Layer (CPU), nr_threads: %d\n", nr_threads);
            for (int run = 0; run < runs; ++run) {
                size_t in_features = dim;
                size_t out_features = dim;
                auto weights = init_wb(dim_t{in_features, out_features});
                auto biases = init_wb(dim_t{out_features});
                auto inputs_q = init_inputs(dim_t{in_features});
                auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                    inputs_q.data(), inputs_q.get_dims());

                auto circuit = new Circuit{new Dense{
                    weights, biases, 5, QuantizationMethod::SimpleQuant, q_const}};
                auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
                int crt_base_size = circuit->infer_crt_base_size(inputs_q);
                auto gc = new GarbledCircuit(circuit, crt_base_size);
                auto g_inputs{gc->garble_inputs(inputs_q)};

                auto t1 = high_resolution_clock::now();
                auto g_outputs{gc->cpu_evaluate(g_inputs, nr_threads)};
                auto t2 = high_resolution_clock::now();
                auto outputs{gc->decode_outputs(g_outputs)};

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "CPU, %d, %d, %lu, %d, %f, %f\n", nr_threads,
                        crt_base_size, dim, run, ms_double.count(), q_acc);

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

void bench_conv2d_cpu(int runs, wandb_t q_const,
                      vector<size_t> dims, FILE* fpt) {
    for (auto dim : dims) {
        for (int nr_threads = 1; nr_threads <= 16; ++nr_threads) {
            printf("Conv2D Layer (CPU), nr_threads: %d\n", nr_threads);

            for (int run = 0; run < runs; ++run) {
                size_t input_width = dim;
                size_t input_height = dim;
                size_t channel = 3;
                // size_t input_size = input_height * input_width * channel;

                size_t filter = 16;
                size_t filter_width = 4;
                size_t filter_height = 4;
                size_t filter_size = filter_width * filter_height;

                size_t stride_width = 2;
                size_t stride_height = 2;

                dim_t weights_dims{filter_width, filter_height, channel,
                                   filter};
                auto weights = init_w_conv2d(weights_dims);
                dim_t bias_dims{filter};
                auto bias = init_b_conv2d(bias_dims, filter_size, channel);
                dim_t input_dims{input_width, input_height, channel};
                auto inputs_q = init_inputs(input_dims);
                auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                    inputs_q.data(), inputs_q.get_dims());

                auto circuit = new Circuit{new Conv2d(
                    weights, bias, input_width, input_height, channel, filter,
                    filter_width, filter_height, stride_width, stride_height,
                    5, QuantizationMethod::SimpleQuant, q_const)};
                auto q_acc = circuit->compute_q_acc(inputs, inputs_q, q_const);
                int crt_base_size = circuit->infer_crt_base_size(inputs_q);
                auto gc = new GarbledCircuit(circuit, crt_base_size);
                auto g_inputs{gc->garble_inputs(inputs_q)};

                auto t1 = high_resolution_clock::now();
                auto g_outputs{gc->cpu_evaluate(g_inputs, nr_threads)};
                auto t2 = high_resolution_clock::now();
                auto outputs{gc->decode_outputs(g_outputs)};

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "CPU, %d, %d, %lu, %d, %f, %f\n", nr_threads,
                        crt_base_size, dim, run, ms_double.count(), q_acc);

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

void bench_approx_relu_cpu(int runs, wandb_t q_const,
                           vector<size_t> dims, FILE* fpt) {
    for (auto dim : dims) {
        vector<float> relu_accs{100.0};
        for (int nr_threads = 1; nr_threads <= 16; ++nr_threads) {
            printf("Approx. ReLU Layer (CPU), nr_threads: %d\n", nr_threads);
            for (auto relu_acc : relu_accs) {
                for (int run = 0; run < runs; ++run) {
                    auto inputs_q = init_inputs(dim_t{dim});
                    auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                        inputs_q.data(), inputs_q.get_dims());

                    auto circuit = new Circuit{new Relu{inputs.get_dims()}};
                    auto q_acc =
                        circuit->compute_q_acc(inputs, inputs_q, q_const);
                    // int crt_base_size = circuit->infer_crt_base_size(inputs);
                    int crt_base_size = 8;

                    auto gc =
                        new GarbledCircuit(circuit, crt_base_size, relu_acc);
                    auto g_inputs{gc->garble_inputs(inputs_q)};
                    auto t1 = high_resolution_clock::now();
                    auto g_outputs{gc->cpu_evaluate(g_inputs, nr_threads)};
                    auto t2 = high_resolution_clock::now();
                    auto outputs{gc->decode_outputs(g_outputs)};

                    duration<double, std::milli> ms_double = t2 - t1;

                    fprintf(fpt, "CPU, %d, %d, %lu, %d, %f, %f, %f\n",
                            nr_threads, crt_base_size, dim, run,
                            ms_double.count(), q_acc, relu_acc);

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
}

void bench_sign_activation_cpu(int runs, wandb_t q_const,
                               vector<size_t> dims, FILE* fpt) {
    for (auto dim : dims) {
        for (int nr_threads = 1; nr_threads <= 16; ++nr_threads) {
            vector<float> sign_accs{100.0};
            printf("Sign Activation Layer (CPU), nr_threads: %d\n", nr_threads);
            for (auto sign_acc : sign_accs) {
                for (int run = 0; run < runs; ++run) {
                    auto inputs_q = init_inputs(dim_t{dim});
                    auto inputs = ScalarTensor<wandb_t>::create_with_cast(
                        inputs_q.data(), inputs_q.get_dims());

                    auto circuit = new Circuit{new Sign{inputs.get_dims()}};
                    auto q_acc =
                        circuit->compute_q_acc(inputs, inputs_q, q_const);
                    // int crt_base_size = circuit->infer_crt_base_size(inputs);
                    int crt_base_size = 8;

                    auto gc =
                        new GarbledCircuit(circuit, crt_base_size, sign_acc);
                    auto g_inputs{gc->garble_inputs(inputs_q)};
                    auto t1 = high_resolution_clock::now();
                    auto g_outputs{gc->cpu_evaluate(g_inputs, nr_threads)};
                    auto t2 = high_resolution_clock::now();
                    auto outputs{gc->decode_outputs(g_outputs)};

                    duration<double, std::milli> ms_double = t2 - t1;

                    fprintf(fpt, "CPU, %d, %d, %lu, %d, %f\n", nr_threads,
                            crt_base_size, dim, run, ms_double.count());

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
}

void bench_rescale_cpu(int runs, wandb_t q_const,
                       vector<size_t> dims, FILE* fpt, bool use_legacy_scaling) {
    for (auto dim : dims) {
        for (int nr_threads = 1; nr_threads <= 16; ++nr_threads) {
            printf("Rescale Layer (CPU), nr_threads: %d\n", nr_threads);
            if(use_legacy_scaling) {
                printf("Using legacy scaling\n");
            } else {
                printf("Using new scaling\n");
            }

            for (int run = 0; run < runs; ++run) {
                auto inputs_q = init_inputs(dim_t{dim});

                Circuit *circuit;
                if(use_legacy_scaling) {
                    circuit = new Circuit{new Rescale{1, inputs_q.get_dims()}};
                } else {
                    const vector<crt_val_t> s = {2};
                    circuit = new Circuit{new Rescale{s, inputs_q.get_dims()}};
                }

                int crt_base_size = 8;

                auto gc = new GarbledCircuit(circuit, crt_base_size, 100.0F);
                auto g_inputs{gc->garble_inputs(inputs_q)};
                auto t1 = high_resolution_clock::now();
                auto g_outputs{gc->cpu_evaluate(g_inputs, nr_threads)};
                auto t2 = high_resolution_clock::now();
                auto outputs{gc->decode_outputs(g_outputs)};
                // outputs.print();

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "CPU, %d, %d, %lu, %d, %f, %d\n", nr_threads,
                        crt_base_size, dim, run, ms_double.count(), use_legacy_scaling);

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

void bench_rescale_scaling_factor_cpu(int runs, wandb_t q_const,
                                      vector<size_t> dims, FILE* fpt) {
    for (auto dim : dims) {
        for (const auto& entry : SCALING_FACTORS_CPM_BASES) {
            const vector<crt_val_t> scaling_factors(entry.second.begin(), entry.second.end());
            printf("Rescale Layer Scaling Factor (CPU), l: %d\n", entry.first);

            for (int run = 0; run < runs; ++run) {
                auto inputs_q = init_inputs(dim_t{dim});
                auto circuit = new Circuit{new Rescale{scaling_factors, inputs_q.get_dims()}};

                int crt_base_size = 8;

                auto gc = new GarbledCircuit(circuit, crt_base_size, 100.0F);
                auto g_inputs{gc->garble_inputs(inputs_q)};
                auto t1 = high_resolution_clock::now();
                auto g_outputs{gc->cpu_evaluate(g_inputs, 16)};
                auto t2 = high_resolution_clock::now();
                auto outputs{gc->decode_outputs(g_outputs)};
                // outputs.print();

                duration<double, std::milli> ms_double = t2 - t1;

                fprintf(fpt, "CPU, %d, %d, %lu, %d, %f, %d\n", 1,
                        crt_base_size, dim, run, ms_double.count(), entry.first);

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

int main() {
    init_cuda();
    FILE* fpt;
    std::string path = "../data/scalability/scalability_";
    auto date_string = get_date_string();

    create_dir("../data");
    create_dir("../data/scalability");

    int runs = 10;
    //
    //
    // Benchmark: Dense Layer
    {
        std::string filename = path + date_string + "_dense.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt,
                "type, nr_threads, crt_base_size, dims, run, runtime, q_acc\n");

        wandb_t q_const = 0.0001;
        vector<size_t> dims{128, 512, 2048};
        // bench_dense_cpu(runs, q_const, dims, fpt);
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
                "type, nr_threads, crt_base_size, dims, run, runtime, q_acc\n");

        wandb_t q_const = 0.0001;
        vector<size_t> dims{64, 128, 256};
        // bench_conv2d_cpu(runs, q_const, dims, fpt);
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
                "type, nr_threads, crt_base_size, dims, run, runtime, q_acc, "
                "relu_acc\n");
        wandb_t q_const = 0.0001;
        vector<size_t> dims{128, 2048, 16384};
        // bench_approx_relu_cpu(runs, q_const, dims, fpt);
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
        fprintf(fpt, "type, nr_threads, crt_base_size, dims, run, runtime\n");
        wandb_t q_const = 0.0001;
        vector<size_t> dims{128, 2048, 16384};
        // bench_sign_activation_cpu(runs, q_const, dims, fpt);
        fclose(fpt);
    }
    //
    //
    //

    //
    //
    // Benchmark: Rescale
    {
        std::string filename = path + date_string + "_rescaling.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt, "type, nr_threads, crt_base_size, dims, run, runtime, use_legacy_scaling\n");
        wandb_t q_const = 0.0001;
        vector<size_t> dims{128};//, 2048, 16384};
        // bench_rescale_cpu(runs, q_const, dims, fpt, true);
        // bench_rescale_cpu(runs, q_const, dims, fpt, false);
        fclose(fpt);
    }
    //
    //
    //

    //
    //
    // Benchmark: Rescale scaling factor
    {
        std::string filename = path + date_string + "_rescaling_scaling_factor.csv";
        fpt = fopen(filename.c_str(), "w+");
        fprintf(fpt, "type, crt_base_size, dims, run, runtime, l\n");
        wandb_t q_const = 0.0001;
        vector<size_t> dims{128};//, 2048, 16384};
        bench_rescale_scaling_factor_cpu(runs, q_const, dims, fpt);
        fclose(fpt);
    }
    //
    //
    //
}

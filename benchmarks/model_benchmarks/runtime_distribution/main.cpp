#define BENCHMARK

#include <chrono>
#include <vector>

#include "../../benchmark_utilities.h"
#include "circuit/circuit.h"
#include "circuit/onnx_modelloader.h"
#include "garbling/garbled_circuit.h"
#include "misc/dataloader.h"
#include "misc/datatypes.h"

using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using std::vector;

void bench_model_cpu(Circuit* circuit,
                     vector<ScalarTensor<q_val_t>> input_images,
                     vector<unsigned long> labels, int nr_inputs,
                     infer_config_t config, FILE* fpt_g_times,
                     FILE* fpt_e_times) {
    printf("RUNTIME DIST: Bench (CPU) Model: %s\n", config.model_name.c_str());

    for (auto relu_acc : config.relu_accs) {
        printf("Relu Accuracy: %f\n", relu_acc);
        vector<unsigned long> infered_labels;
        for (int i = 0; i < nr_inputs; ++i) {
            auto inputs = input_images.at(i);

            // std::cout << "Garbling circuit..." << std::endl;

            GarbledCircuit* gc;
            if(config.optimize_bases) {
                std::vector<crt_val_t> opt_crt;
                std::vector<mrs_val_t> opt_mrs;

                if(config.model_name == "MODEL_F_GNNP_POOL_REPL") {
                    opt_crt = {32, 53, 59};
                    opt_mrs = {22, 21, 21, 19};
                    std::cout << "Using custom bases for model: " << config.model_name << std::endl;
                }
                else if(config.model_name == "MODEL_F_MINIONN_POOL_REPL") {
                    opt_crt = {32, 97, 107};
                    opt_mrs = {22, 19, 15, 13};
                    std::cout << "Using custom bases for model: " << config.model_name << std::endl;
                }
                else {
                    throw std::runtime_error("Unknown model name for optimized bases.");
                }

                const auto max_modulus = opt_crt.back();
                
                gc = new GarbledCircuit(circuit, opt_crt, opt_mrs, max_modulus);
            }
            else {
                gc = new GarbledCircuit(circuit, config.target_crt_base_size, relu_acc);
            }

            auto g_inputs{gc->garble_inputs(inputs)};

            auto g_outputs{gc->cpu_evaluate(g_inputs)};
            auto outputs{gc->decode_outputs(g_outputs)};

            auto infered_label = outputs.argmax();
            infered_labels.push_back(infered_label);

            auto garbling_times = gc->get_garbling_times();
            for (auto const& [key, val] : garbling_times) {
                fprintf(fpt_g_times, "CPU, %s, %s, %d, %d, %f, %f\n",
                        config.model_name.c_str(), key.c_str(),
                        config.target_crt_base_size, config.optimize_bases, val, relu_acc);
            }

            auto evaluation_times = gc->get_evaluation_times();
            for (auto const& [key, val] : evaluation_times) {
                fprintf(fpt_e_times, "CPU, %s, %s, %d, %d, %f, %f\n",
                        config.model_name.c_str(), key.c_str(),
                        // config.target_crt_base_size, config.optimize_bases, val, relu_acc);
                        config.target_crt_base_size, 0, val, relu_acc);
            }

            //// clean up
            for (auto label : *g_inputs) {
                delete label;
            }
            delete g_inputs;
            delete gc;
        }

        // compute prediction-accuracy
        int correct = 0;
        for (size_t i = 0; i < infered_labels.size(); i++) {
            if (infered_labels.at(i) == labels.at(i)) {
                correct++;
            }
        }
        double accuracy = (double)correct / (double)infered_labels.size();

        printf("Accuracy: %f\n", accuracy);
    }
}

//TODO: port to redash
// void bench_model_gpu(Circuit* circuit,
//                      vector<ScalarTensor<q_val_t>> input_images,
//                      vector<unsigned long> labels, int nr_inputs,
//                      infer_config_t config, FILE* fpt_e_times) {
//     printf("Bench (GPU) Model: %s\n", config.model_name.c_str());

//     for (auto relu_acc : config.relu_accs) {
//         printf("Relu Accuracy: %f\n", relu_acc);
//         vector<unsigned long> infered_labels;
//         for (int i = 0; i < nr_inputs; ++i) {
//             auto inputs = input_images.at(i);
//             auto gc = new GarbledCircuit(circuit, config.target_crt_base_size,
//                                          relu_acc);
//             gc->cuda_move();

//             auto g_inputs{gc->garble_inputs(inputs)};
//             auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
//             gc->cuda_evaluate(g_dev_inputs);
//             auto g_outputs{gc->cuda_move_outputs()};
//             auto outputs{gc->decode_outputs(g_outputs)};

//             auto infered_label = outputs.argmax();
//             infered_labels.push_back(infered_label);

//             auto evaluation_times = gc->get_evaluation_times();
//             for (auto const& [key, val] : evaluation_times) {
//                 fprintf(fpt_e_times, "GPU, %s, %s, %d, %f, %f\n",
//                         config.model_name.c_str(), key.c_str(),
//                         config.target_crt_base_size, val, relu_acc);
//             }

//             //// clean up
//             for (auto label : *g_inputs) {
//                 delete label;
//             }
//             delete g_inputs;
//             gc->cuda_free_inputs(g_dev_inputs);
//             delete gc;
//         }

//         // compute prediction-accuracy
//         int correct = 0;
//         for (size_t i = 0; i < infered_labels.size(); i++) {
//             if (infered_labels.at(i) == labels.at(i)) {
//                 correct++;
//             }
//         }
//         double accuracy = (double)correct / (double)infered_labels.size();

//         printf("Accuracy: %f\n", accuracy);
//     }
// }

int main() {
    init_cuda();
    FILE* fpt_g_times;
    FILE* fpt_e_times;
    std::string path = "../data/";
    auto date_string = get_date_string();

    create_dir(path);

    //
    //
    // Benchmark
    std::string filename =
        path + date_string + "_runtime_distribution_garbling.csv";
    fpt_g_times = fopen(filename.c_str(), "w+");
    fprintf(fpt_g_times,
            "type, model, layer, target_crt_base_size, optimize_bases, runtime, relu_acc\n");

    std::string filename2 =
        path + date_string + "_runtime_distribution_evaluation.csv";
    fpt_e_times = fopen(filename2.c_str(), "w+");
    fprintf(fpt_e_times,
            "type, model, layer, target_crt_base_size, optimize_bases, runtime, relu_acc\n");

    int nr_inputs = 2;
    auto mnist_dataset = mnist("../../../data/MNIST/raw");
    auto cifar10_dataset =
        cifar10("../../../data/cifar10/cifar-10-batches-bin");
    std::string model_dir = "../../../models/";

    vector<infer_config_t> configs;

    infer_config_t MODEL_A_config{
        .target_crt_base_size = 8,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_A",
        .model_file = "MODEL_A",
        .quantization_method = QuantizationMethod::SimpleQuant,
        .q_parameter =  -1,
        .optimize_bases = false};
    configs.push_back(MODEL_A_config);

    infer_config_t MODEL_B_POOL_REPL_config{
        .target_crt_base_size = 9,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_B",
        .model_file = "MODEL_B",
        .quantization_method = QuantizationMethod::SimpleQuant,
        .q_parameter =  -1,
        .optimize_bases = false};
    configs.push_back(MODEL_B_POOL_REPL_config);

    infer_config_t MODEL_C_config{
        .target_crt_base_size = 9,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_C",
        .model_file = "MODEL_C",
        .quantization_method = QuantizationMethod::SimpleQuant,
        .q_parameter =  -1,
        .optimize_bases = false};
    configs.push_back(MODEL_C_config);

    infer_config_t MODEL_D_POOL_REPL_config{
        .target_crt_base_size = 8,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_D",
        .model_file = "MODEL_D",
        .quantization_method = QuantizationMethod::SimpleQuant,
        .q_parameter =  -1,
        .optimize_bases = false};
    configs.push_back(MODEL_D_POOL_REPL_config);

    infer_config_t MODEL_F_GNNP_POOL_REPL_config_OPT{
        .target_crt_base_size = 7,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = cifar10_dataset,
        .model_name = "MODEL_F_GNNP_POOL_REPL",
        .model_file = "MODEL_F_GNNP_POOL_REPL",
        .quantization_method = QuantizationMethod::ScaleQuantPlus,
        .q_parameter = 32,
        .optimize_bases = true};
    configs.push_back(MODEL_F_GNNP_POOL_REPL_config_OPT);

    infer_config_t MODEL_F_GNNP_POOL_REPL_config_CPM{
        .target_crt_base_size = 7,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = cifar10_dataset,
        .model_name = "MODEL_F_GNNP_POOL_REPL",
        .model_file = "MODEL_F_GNNP_POOL_REPL",
        .quantization_method = QuantizationMethod::ScaleQuant,
        .q_parameter =  6,
        .optimize_bases = false};
    configs.push_back(MODEL_F_GNNP_POOL_REPL_config_CPM);

    infer_config_t MODEL_F_MINIONN_POOL_REPL_config_OPT{
        .target_crt_base_size = 7,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = cifar10_dataset,
        .model_name = "MODEL_F_MINIONN_POOL_REPL",
        .model_file = "MODEL_F_MINIONN_POOL_REPL",
        .quantization_method = QuantizationMethod::ScaleQuantPlus,
        .q_parameter =  32,
        .optimize_bases = true};
    configs.push_back(MODEL_F_MINIONN_POOL_REPL_config_OPT);

    infer_config_t MODEL_F_MINIONN_POOL_REPL_config_CPM{
        .target_crt_base_size = 7,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = cifar10_dataset,
        .model_name = "MODEL_F_MINIONN_POOL_REPL",
        .model_file = "MODEL_F_MINIONN_POOL_REPL",
        .quantization_method = QuantizationMethod::ScaleQuant,
        .q_parameter =  6,
        .optimize_bases = false};
    configs.push_back(MODEL_F_MINIONN_POOL_REPL_config_CPM);

    for (size_t i = 0; i < configs.size(); ++i) {
        auto config = configs.at(i);
        std::string model_path = model_dir + config.model_file + ".onnx";

        Circuit* circuit =
            load_onnx_model(model_path, config.quantization_method, config.q_parameter);

        vector<ScalarTensor<wandb_t>> small_test_images;
        vector<unsigned long> small_test_labels;
        for (int i = 0; i < nr_inputs; ++i) {
            auto tmp_image{config.dataset.testing_images.at(i)};
            auto tmp_label{config.dataset.testing_labels.at(i)};
            small_test_images.push_back(tmp_image);
            small_test_labels.push_back(tmp_label);
        }

        vector<ScalarTensor<q_val_t>> small_test_images_q;
        if (config.quantization_method == QuantizationMethod::SimpleQuant) {
            circuit->optimize_quantization(config.target_crt_base_size,
                                           small_test_images, 0.25, 0.01,
                                           0.0001);
            wandb_t q_const = circuit->get_q_const();
            small_test_images_q = quantize(small_test_images,
                                           config.quantization_method, q_const);
        } else if (config.quantization_method == QuantizationMethod::ScaleQuant) {
            small_test_images_q = quantize(small_test_images, config.quantization_method, 1 << circuit->get_q_parameter());
        } else { // ScaleQuantPlus
            small_test_images_q = quantize(small_test_images, config.quantization_method, circuit->get_q_parameter());
        }

        bench_model_cpu(circuit, small_test_images_q, small_test_labels,
                        nr_inputs, config, fpt_g_times, fpt_e_times);
        // bench_model_gpu(circuit, small_test_images_q, small_test_labels,
        //                 nr_inputs, config, fpt_e_times);

        delete circuit;
    }
    fclose(fpt_g_times);
    fclose(fpt_e_times);
    //
    //
    //
}
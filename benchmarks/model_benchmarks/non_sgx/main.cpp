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

void bench_model_cpu(Circuit* circuit,
                     vector<ScalarTensor<q_val_t>> input_images,
                     vector<unsigned long> labels, int nr_inputs,
                     std::string model_dir, infer_config_t config, FILE* fpt) {
    printf("Bench (CPU) Model: %s\n", config.model_name.c_str());

    for (auto relu_acc : config.relu_accs) {
        printf("Relu Accuracy: %f\n", relu_acc);
        vector<unsigned long> infered_labels;
        for (int i = 0; i < nr_inputs; ++i) {
            auto inputs = input_images.at(i);
            auto gc = new GarbledCircuit(circuit, config.target_crt_base_size,
                                         relu_acc);

            auto g_inputs{gc->garble_inputs(inputs)};

            auto t1 = high_resolution_clock::now();
            auto g_outputs{gc->cpu_evaluate(g_inputs)};
            auto outputs{gc->decode_outputs(g_outputs)};
            auto t2 = high_resolution_clock::now();

            auto infered_label = outputs.argmax();
            infered_labels.push_back(infered_label);

            duration<double, std::milli> ms_double = t2 - t1;
            auto label = labels.at(i);

            fprintf(fpt, "CPU, %s, %d, %f, %f, %ld, %ld\n",
                    config.model_name.c_str(), config.target_crt_base_size,
                    ms_double.count(), relu_acc, label, infered_label);

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

void bench_model_gpu(Circuit* circuit,
                     vector<ScalarTensor<q_val_t>> input_images,
                     vector<unsigned long> labels, int nr_inputs,
                     std::string model_dir, infer_config_t config, FILE* fpt) {
    printf("Bench (GPU) Model: %s\n", config.model_name.c_str());

    std::string model_path = model_dir + config.model_name + ".onnx";

    for (auto relu_acc : config.relu_accs) {
        printf("Relu Accuracy: %f\n", relu_acc);
        vector<unsigned long> infered_labels;
        for (int i = 0; i < nr_inputs; ++i) {
            auto inputs = input_images.at(i);
            auto gc = new GarbledCircuit(circuit, config.target_crt_base_size,
                                         relu_acc);
            gc->cuda_move();

            auto t1 = high_resolution_clock::now();
            auto g_inputs{gc->garble_inputs(inputs)};
            auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
            gc->cuda_evaluate(g_dev_inputs);
            auto g_outputs{gc->cuda_move_outputs()};
            auto outputs{gc->decode_outputs(g_outputs)};
            auto t2 = high_resolution_clock::now();

            auto infered_label = outputs.argmax();
            infered_labels.push_back(infered_label);

            duration<double, std::milli> ms_double = t2 - t1;
            auto label = labels.at(i);

            fprintf(fpt, "GPU, %s, %d, %f, %f, %ld, %ld\n",
                    config.model_name.c_str(), config.target_crt_base_size,
                    ms_double.count(), relu_acc, label, infered_label);

            //// clean up
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
            if (infered_labels.at(i) == labels.at(i)) {
                correct++;
            }
        }
        double accuracy = (double)correct / (double)infered_labels.size();

        printf("Accuracy: %f\n", accuracy);
    }
}

int main() {
    init_cuda();
    FILE* fpt;
    // FILE* fpt2;
    std::string path = "../data/";
    auto date_string = get_date_string();

    create_dir(path);

    //
    //
    // Benchmark
    std::string filename = path + date_string + "_plain_models.csv";
    fpt = fopen(filename.c_str(), "w+");
    fprintf(fpt,
            "type, model, target_crt_base_size, runtime, relu_acc, "
            "label, infered_label\n");

    // filename = path + date_string + "_plain_models.csv";
    // fpt2 = fopen(filename.c_str(), "w+");
    // fprintf(
    //     fpt2,
    //     "model_name, plain_acc, plain_q_acc, q_const,
    //     target_crt_base_size\n");

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
        .quantization_method = QuantizationMethod::SimpleQuant};
    configs.push_back(MODEL_A_config);

    infer_config_t MODEL_B_POOL_REPL_config{
        .target_crt_base_size = 9,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_B",
        .model_file = "MODEL_B_POOL_REPL",
        .quantization_method = QuantizationMethod::SimpleQuant};
    configs.push_back(MODEL_B_POOL_REPL_config);

    infer_config_t MODEL_C{
        .target_crt_base_size = 9,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_C",
        .model_file = "MODEL_C",
        .quantization_method = QuantizationMethod::SimpleQuant};
    configs.push_back(MODEL_C);

    infer_config_t MODEL_D_POOL_REPL_config{
        .target_crt_base_size = 8,
        .relu_accs = {100.0},  //{100.0, 99.999, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_D",
        .model_file = "MODEL_D_POOL_REPL",
        .quantization_method = QuantizationMethod::SimpleQuant};
    configs.push_back(MODEL_D_POOL_REPL_config);

    infer_config_t MODEL_E_30_config{
        .target_crt_base_size = 5,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_E_30",
        .model_file = "MODEL_E_30",
        .quantization_method = QuantizationMethod::SimpleQuant};
    configs.push_back(MODEL_E_30_config);

    infer_config_t MODEL_E_100_config{
        .target_crt_base_size = 5,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = mnist_dataset,
        .model_name = "MODEL_E_100",
        .model_file = "MODEL_E_100",
        .quantization_method = QuantizationMethod::SimpleQuant};
    configs.push_back(MODEL_E_100_config);

    infer_config_t MODEL_F_GNNP_POOL_REPL_config{
        .target_crt_base_size = 7,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = cifar10_dataset,
        .model_name = "MODEL_F_GNNP_POOL_REPL",
        .model_file = "MODEL_F_GNNP_POOL_REPL",
        .quantization_method = QuantizationMethod::ScaleQuant};
    configs.push_back(MODEL_F_GNNP_POOL_REPL_config);

    infer_config_t MODEL_F_MINIONN_POOL_REPL_config{
        .target_crt_base_size = 7,
        .relu_accs = {100.0},  //{100.0, 99.99, 99.9, 99.0},
        .dataset = cifar10_dataset,
        .model_name = "MODEL_F_MINIONN_POOL_REPL",
        .model_file = "MODEL_F_MINIONN_POOL_REPL",
        .quantization_method = QuantizationMethod::ScaleQuant};
    configs.push_back(MODEL_F_MINIONN_POOL_REPL_config);

    for (size_t i = 0; i < configs.size(); ++i) {
        auto config = configs.at(i);
        std::string model_path = model_dir + config.model_file + ".onnx";
        Circuit* circuit =
            load_onnx_model(model_path, config.quantization_method);

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
            small_test_images_q = quantize(small_test_images, config.quantization_method, q_const);
        } else {
            small_test_images_q = quantize(small_test_images, config.quantization_method, QL);
        }

        // auto plain_test_acc = circuit->plain_test(
        //     config.dataset.testing_images, config.dataset.testing_labels);
        // auto plain_q_test_acc =
        // circuit->plain_q_test(dataset_q.testing_images,
        //                                               dataset_q.testing_labels);

        // fprintf(fpt2, "%s, %f, %f, %f, %d\n", config.model_name.c_str(),
        //         plain_test_acc, plain_q_test_acc, q_const,
        //         config.target_crt_base_size);

        bench_model_cpu(circuit, small_test_images_q, small_test_labels,
                        nr_inputs, model_dir, config, fpt);
        bench_model_gpu(circuit, small_test_images_q, small_test_labels,
                        nr_inputs, model_dir, config, fpt);

        delete circuit;
    }
    // fclose(fpt2);
    fclose(fpt);
    //
    //
    //
}
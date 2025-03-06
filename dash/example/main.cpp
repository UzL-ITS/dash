// This is a non-sgx main for testing purposes.
#include <random>
#include <vector>
#include <chrono>

#include "circuit/circuit.h"
#include "circuit/layer/conv2d.h"
#include "circuit/layer/dense.h"
#include "circuit/layer/flatten.h"
#include "circuit/layer/layer.h"
#include "circuit/layer/max_pool2d.h"
#include "circuit/layer/mixed_mod_mult_layer.h"
#include "circuit/layer/mult_layer.h"
#include "circuit/layer/projection.h"
#include "circuit/layer/relu.h"
#include "circuit/layer/rescale.h"
#include "circuit/layer/max.h"
#include "circuit/layer/base_extension.h"
#include "circuit/onnx_modelloader.h"
#include "garbling/gadgets/lookup_approx_sign.h"
#include "garbling/gadgets/base_extension_gadget.h"
#include "garbling/garbled_circuit.h"
#include "garbling/label_tensor.h"
#include "garbling/layer/garbled_dense.h"
#include "misc/cuda_util.h"
#include "misc/dataloader.h"
#include "misc/misc.h"
#include "misc/util.h"

using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::microseconds;


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

using std::vector;

crt_val_t test_proj_function(crt_val_t x, void* params) { return x / 2; }
crt_val_t test_proj_function2(crt_val_t x, void* params) { return x + 1; }

__host__ __device__ void print_hex(unsigned char* data, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

void check_relu_correctness(ScalarTensor<q_val_t>& outputs, int crt_modulus) {
    int false_count = 0;
    for (int i = 0; i < crt_modulus / 2; ++i) {
        if (outputs.data()[i] != 0) {
            false_count++;
        }
    }

    for (int i = crt_modulus / 2; i < crt_modulus; ++i) {
        if (outputs.data()[i] != i - crt_modulus / 2) {
            false_count++;
        }
    }

    double corrrectness = (crt_modulus - false_count) / (double)crt_modulus;

    std::cout << "correctness: " << corrrectness << std::endl;
}

int main() {
    init_cuda();

#ifdef LABEL_TENSOR_USE_EIGEN
    std::cerr << " * * * * EIGEN ENABLED * * * * " << std::endl;
#endif

    //
    //
    // Example-Code: Felix Herumprobieren
    // size_t IN_SIZE = 3*3;
    // constexpr size_t OUT_SIZE = 5;
    // constexpr wandb_t Q_CONST = 1;

    //CIRCUITS
    //util::get random vector seg fault wenn min > max
    // const ScalarTensor<wandb_t> weights { util::get_random_vector<wandb_t>(OUT_SIZE * IN_SIZE, -5, 5), dim_t{OUT_SIZE, IN_SIZE} };
    // const ScalarTensor<wandb_t> biases { util::get_random_vector<wandb_t>(OUT_SIZE, 0, 1), dim_t{OUT_SIZE} };
    // const auto circuit = new Circuit{ 
    //     new Dense{weights, biases, QuantizationMethod::SimpleQuant,Q_CONST},
    //     //new Sign{dim_t{OUT_SIZE}},
    //     new Relu{dim_t{OUT_SIZE}}
    //     };
    // // const vector<crt_val_t> crt_base{2, 3, 5};
    // // const vector<mrs_val_t> mrs_base{26, 6, 3, 2};
    // const auto garbledCircuit = new GarbledCircuit{ circuit, 4, 100.0f};

    // //INPUT
    // const ScalarTensor<q_val_t> inputs { util::get_random_vector<q_val_t>(IN_SIZE, 0, 5), dim_t{IN_SIZE} };

    // //EVALUATE
    // const auto garbledInputs{garbledCircuit->garble_inputs(inputs)};
    // const auto garbledOutputs = garbledCircuit->cpu_evaluate(garbledInputs);
    // const auto outputs{garbledCircuit->decode_outputs(garbledOutputs)};

    // std::cout << "weights: " << std::endl;
    // weights.print();
    // std::cout << "biases: " << std::endl;
    // biases.print();
    // std::cout << "inputs: " << std::endl;
    // inputs.print();
    // std::cout << "garbled outputs: " << std::endl;
    // outputs.print();

    // //CHECK 
    // const auto check = circuit->plain_q_eval(inputs);
    // std::cout << "plain outputs: " << std::endl;
    // check.print();

    // //FREE
    // for (auto label : *garbledInputs) delete label;
    // delete garbledInputs;
    // delete circuit;
    // delete garbledCircuit;

    // std::cout << "DONE FELIX" << std::endl;

    //
    //
    // Example-Code: Dense Layer
    // size_t input_size = 512 * 512;
    // size_t output_size = 10;
    // wandb_t q_const = 1;
    // const wandb_t min = 0;
    // const wandb_t max = 100;

    // vector<wandb_t> rand = util::get_random_vector<wandb_t>(output_size * input_size, min, max);
    // ScalarTensor<wandb_t> weights{rand, dim_t{output_size, input_size}};

    // rand = util::get_random_vector<wandb_t>(output_size, min, max);
    // ScalarTensor<wandb_t> biases{rand, dim_t{output_size}};

    // vector<q_val_t> rand_q = util::get_random_vector<q_val_t>(input_size, min, max); 
    // ScalarTensor<q_val_t> inputs{rand_q, dim_t{input_size}};

    // auto circuit = new Circuit{
    //     new Dense{weights, biases, QuantizationMethod::SimpleQuant,
    //     q_const}};

    // auto gc = new GarbledCircuit(circuit, 8);
    // auto g_inputs{gc->garble_inputs(inputs)};
    // auto g_outputs = gc->cpu_evaluate(g_inputs);
    // auto outputs{gc->decode_outputs(g_outputs)};

    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // auto check = circuit->plain_q_eval(inputs);

    // std::cout << "check: " << std::endl;
    // check.print();

    // assert(outputs == check && "Dense Layer: outputs != check");

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;

    // delete circuit;
    // delete gc;
    //

    //
    //
    // Example-Code: Conv2D Layer
    // Conv parameters
    // wandb_t q_const = 1;
    // size_t input_width = 5;
    // size_t input_height = 5;
    // size_t input_size = input_height * input_width;
    // size_t channel = 3;

    // size_t filter = 1;
    // size_t filter_width = 2;
    // size_t filter_height = 2;
    // size_t filter_size = filter_width * filter_height;

    // size_t stride_width = 1;
    // size_t stride_height = 1;

    // // Generate example data
    // int size = input_size * channel * sizeof(q_val_t);
    // q_val_t* inputs = (q_val_t*)malloc(size);
    // size = filter_size * channel * filter * sizeof(wandb_t);
    // wandb_t* weights = (wandb_t*)malloc(size);
    // wandb_t* bias = (wandb_t*)malloc(filter * sizeof(wandb_t));

    // for (int i = 0; i < input_size * channel; ++i) {
    //     inputs[i] = i;
    //     // inputs[i] = i >= 2 * input_size ? 1 : 0;
    // }

    // // bool flip = false;
    // for (int i = 0; i < filter_size * channel * filter; ++i) {
    //     weights[i] = 1;
    //     // flip = !flip;
    // }

    // for (int i = 0; i < filter; ++i) {
    //     bias[i] = 0;  //(i*10)%100;
    // }

    // // print inputs, weights and bias
    // std::cout << "inputs: " << std::endl;
    // for (int i = 0; i < input_size * channel; ++i) {
    //     std::cout << inputs[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "weights: " << std::endl;
    // for (int i = 0; i < filter_size * channel * filter; ++i) {
    //     std::cout << weights[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "bias: " << std::endl;
    // for (int i = 0; i < filter; ++i) {
    //     std::cout << bias[i] << " ";
    // }

    // dim_t input_dims{input_width, input_height, channel};
    // ScalarTensor<q_val_t> inputs_t{inputs, input_dims};

    // dim_t weights_dims{filter_width, filter_height, channel, filter};
    // ScalarTensor<wandb_t> weights_t{weights, weights_dims};

    // dim_t bias_dims{filter};
    // ScalarTensor<wandb_t> bias_t{bias, bias_dims};

    // auto circuit = new Circuit{new Conv2d(
    //     weights_t, bias_t, input_width, input_height, channel, filter,
    //     filter_width, filter_height, stride_width, stride_height, QuantizationMethod::SimpleQuant, q_const)};

    // // auto acc = circuit->compute_q_acc(inputs_t, 1, q_const);
    // // std::cout << "accuracy: " << acc << std::endl;

    // auto gc = new GarbledCircuit(circuit, 8);
    // // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(inputs_t)};
    // // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // // gc->cuda_evaluate(g_dev_inputs);
    // // auto g_outputs{gc->cuda_move_outputs()};
    // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // auto outputs{gc->decode_outputs(g_outputs)};

    // std::cout << "output_dims: ";
    // for (auto dim : outputs.get_dims()) {
    //     std::cout << dim << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // free(inputs);
    // free(weights);
    // free(bias);

    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    // 
    //
    // Example-Code: im2col and im2row on LabelTensor with unique labels
    // crt_val_t modulus = 17;
    // size_t channels = 3;
    // size_t nr_comps = LabelTensor::get_nr_comps(modulus);
    // vector<crt_val_t> values(3 * 3 * channels * nr_comps);
    // for (size_t i = 0; i < 3 * 3 * channels; ++i) {
    //     std::fill(values.begin() + i * nr_comps, values.begin() + (i + 1) * nr_comps, i);
    // }
    // LabelTensor tensor{modulus, values, dim_t{3, 3, channels}};
    
    // //print entire tensor
    // tensor.print();

    // auto im2col_result = tensor.im2col(2, 2, 1, 1);
    // // print im2col result in its correct dimensions
    // std::cout << "im2col Result: " << std::endl;
    // std::cout << "dims: " << im2col_result.get_dims()[0] << " " << im2col_result.get_dims()[1] << " " << std::endl;
    // im2col_result.print();

    // auto im2row_result = tensor.im2row(2, 2, 1, 1);
    // std::cout << "im2row Result: " << std::endl;
    // std::cout << "dims: " << im2row_result.get_dims()[0] << " " << im2row_result.get_dims()[1] << " " << std::endl;
    // im2row_result.print();
    //
    //
    //
    
    //Example-Code: Max Layer
    // ScalarTensor<q_val_t> inputs{{3, 3}, dim_t{2}};

    // auto circuit = new Circuit{new Max()};
    // std::cout << "main: inputs: " << inputs.size() << std::endl;

    // vector<crt_val_t> crt_base{2, 3, 5};
    // vector<mrs_val_t> mrs_base{26, 6, 3, 2};

    // auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);
    // //gc->cuda_move();

    // auto g_inputs{gc->garble_inputs(inputs)};
    // std::cout << "main: garbled inputs: " << g_inputs->at(0)->get_nr_label() << std::endl;

    // //auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // //gc->cuda_evaluate(g_dev_inputs);
    // //auto g_outputs{gc->cuda_move_outputs()};

    // auto outputs{gc->decode_outputs(g_outputs)};
    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // //gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Example-Code: Max-Pooling 2D Layer
    // ScalarTensor<q_val_t> inputs{{{1, 2, 3, 4, 5 , -4, -3, -2, -1}}, dim_t{3, 3, 1}};

    // auto circuit = new Circuit{new MaxPool2d(3, 3, 1, 3, 3)};
    // std::cout << "main: input count: " << inputs.size() << std::endl;

    // vector<crt_val_t> crt_base{2, 3, 5};
    // vector<mrs_val_t> mrs_base{26, 6, 3, 2};

    // auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);
    // //gc->cuda_move();

    // auto g_inputs{gc->garble_inputs(inputs)};
    // std::cout << "main: garbled input count: " << g_inputs->at(0)->get_nr_label() << std::endl;

    // //auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // //gc->cuda_evaluate(g_dev_inputs);
    // //auto g_outputs{gc->cuda_move_outputs()};

    // auto outputs{gc->decode_outputs(g_outputs)};
    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // //gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Example-Code: Mult Layer
    // ScalarTensor<q_val_t> inputs{{2, 3, 4, 2}, dim_t{2, 2}};

    // dim_t output_dims{2};
    // auto circuit = new Circuit{new MultLayer{inputs.get_dims(),
    // output_dims}};

    // vector<crt_val_t> crt_base{19};

    // auto gc = new GarbledCircuit(circuit, crt_base);
    // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(inputs)};

    // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // gc->cuda_evaluate(g_dev_inputs);
    // auto g_outputs{gc->cuda_move_outputs()};

    // auto outputs{gc->decode_outputs(g_outputs)};
    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // gc->cuda_free_inputs(g_dev_inputs);
    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Example-Code: MixedModMult Layer
    // ScalarTensor<q_val_t> inputs{{2, 1, 3, 0}, dim_t{2, 2}};

    // dim_t output_dims{2};
    // auto circuit =
    //     new Circuit{new MixedModMultLayer{inputs.get_dims(), output_dims,
    //     2}};

    // vector<crt_val_t> crt_base{19};

    // auto gc = new GarbledCircuit(circuit, crt_base);
    // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(inputs)};

    // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // gc->cuda_evaluate(g_dev_inputs);
    // auto g_outputs{gc->cuda_move_outputs()};
    // auto outputs{gc->decode_outputs(g_outputs)};
    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Example-Code: LookupApproxSign (Lookup computation for approximated sign)
    // vector<crt_val_t> crt_base{19};
    // vector<mrs_val_t> mrs_base{25, 4};
    // auto lookup = new LookupApproxSign(crt_base, mrs_base);

    // vector<crt_val_t> crt_value{2};

    // std::cout << "approximation in mrs:\n";
    // for (size_t i = 0; i < crt_base.size(); ++i) {
    //     for (size_t j = 0; j < mrs_base.size(); ++j)
    //         std::cout << lookup->get(i, crt_value[i])[j]
    //                   << " ";
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // // cleanup
    // delete lookup;
    //
    //
    //

    //
    //
    // Example-Code: Relu Layer
    // ScalarTensor<q_val_t> inputs{{1, 0, -1}, dim_t{3}};

    // auto circuit = new Circuit{new Relu{inputs.get_dims()}};

    // vector<crt_val_t> crt_base{2, 3, 5};
    // vector<mrs_val_t> mrs_base{26, 6, 3, 2};

    // auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);
    // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(inputs)};
    // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // gc->cuda_evaluate(g_dev_inputs);
    // auto g_outputs{gc->cuda_move_outputs()};

    // auto outputs{gc->decode_outputs(g_outputs)};
    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Example-Code: Flatten Layer
    // ScalarTensor<q_val_t> inputs{1, dim_t{3, 3}};

    // auto output_dims = inputs.get_dims();
    // auto circuit = new Circuit{new Flatten{inputs.get_dims()}};

    // vector<crt_val_t> crt_base{2, 3, 5};
    // vector<mrs_val_t> mrs_base{26, 6, 3, 2};

    // auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);

    // auto g_inputs{gc->garble_inputs(inputs)};

    // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    // auto outputs{gc->decode_outputs(g_outputs)};
    // std::cout << "outputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Example-Code: End-To-End
    // Load the data
    // std::cout.precision(17);

    // auto dataset = mnist("../../data/MNIST/raw");
    // for (int i = 0; i < 28; i++) {
    //     for (int j = 0; j < 28; j++) {
    //         int o = 0;
    //         if (dataset.training_images.at(1).at(i * 28 + j) > 0) {
    //             o = 1;
    //         }
    //         std::cout << o << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // auto inputs = dataset.training_images.at(1);

    // Circuit* circuit =
    //     load_onnx_model("../../models/trained_models/MODEL_A.onnx", 0.05);

    // auto plain_outputs = circuit->plain_eval(inputs, 255.0);

    // std::cout << "plain outputs: " << std::endl;
    // plain_outputs.print();

    // auto plain_q_outputs = circuit->plain_q_eval(inputs);
    // std::cout << "\nplain quantized outputs: " << std::endl;
    // plain_q_outputs.print();

    // for (auto l : circuit->get_layer()) {
    //     std::cout << "min: " << l->get_min_plain_q_val() << std::endl;
    //     std::cout << "max: " << l->get_max_plain_q_val() << std::endl;
    //     std::cout << std::endl;
    // }

    // // you can also use the whole dataset.training_images vector
    // int crt_base_size =
    //     circuit->infer_crt_base_size(dataset.training_images.at(0));

    // std::cout << "crt_base_size: " << crt_base_size << std::endl;

    // auto gc = new GarbledCircuit(circuit, crt_base_size, (float)100.0);
    // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(inputs)};
    // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // gc->cuda_evaluate(g_dev_inputs);
    // auto g_outputs{gc->cuda_move_outputs()};
    // auto outputs{gc->decode_outputs(g_outputs)};

    // std::cout << "output_label: " << outputs.argmax() << std::endl;

    // std::cout << "\noutputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // ReLU Correctness Check
    // for (int i = 4; i <= 5; ++i) {
    //     vector<crt_val_t> crt_base =
    //     util::sieve_of_eratosthenes<crt_val_t>(i); int crt_modulus =
    //     std::accumulate(crt_base.begin(), crt_base.end(), 1,
    //                                       std::multiplies<crt_val_t>());
    //     vector<q_val_t> in(crt_modulus);
    //     iota(in.begin(), in.end(), -crt_modulus / 2);

    //     ScalarTensor<q_val_t> inputs{in, dim_t{crt_modulus}};

    //     auto circuit = new Circuit{new Relu{inputs.get_dims()}};
    //     auto gc = new GarbledCircuit(circuit, i, 100.0F);
    //     gc->cuda_move();
    //     auto g_inputs{gc->garble_inputs(inputs)};
    //     auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    //     // auto g_outputs{gc->cpu_evaluate(g_inputs)};
    //     gc->cuda_evaluate(g_dev_inputs);
    //     auto g_outputs{gc->cuda_move_outputs()};
    //     auto outputs{gc->decode_outputs(g_outputs)};

    //     std::cout << "crt_base_size: " << i << std::endl;
    //     std::cout << "crt_base: ";
    //     for (auto base : crt_base) {
    //         std::cout << base << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "mrs_base: ";
    //     for (auto base : gc->get_mrs_base()) {
    //         std::cout << base << " ";
    //     }
    //     std::cout << std::endl;
    //     check_relu_correctness(outputs, crt_modulus);

    //     // clean up
    //     for (auto label : *g_inputs) {
    //         delete label;
    //     }
    //     delete g_inputs;
    //     gc->cuda_free_inputs(g_dev_inputs);

    //     delete circuit;
    //     delete gc;
    // }
    //
    //
    //

    //
    //
    // Add sign-layer to garbling
    // ScalarTensor<q_val_t> plain_inputs{{-2, -1, 0, 1, 2}, dim_t{5}};
    // printf("plain inputs:\n");
    // plain_inputs.print();
    // auto circuit = new Circuit{new Sign{plain_inputs.get_dims()}};
    // auto plain_outputs = circuit->plain_q_eval(plain_inputs);
    // printf("plain output:\n");
    // plain_outputs.print();

    // auto gc = new GarbledCircuit(circuit, 7, (float)100.0);
    // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(plain_inputs)};
    // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // // auto g_output{gc->cpu_evaluate(g_inputs)};
    // gc->cuda_evaluate(g_dev_inputs);
    // auto g_output{gc->cuda_move_outputs()};
    // auto outputs{gc->decode_outputs(g_output)};
    // printf("outputs:\n");
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    // Add Rescaling Layer // TODO: this example is still slightly off... adding another factor to s creates all-0 result, which is wrong too. I think a redesign of the swapping in BE should help
    // const auto prod = 11 * 13 * 17;
    // vector<q_val_t> in(prod);
    // iota(in.begin(), in.end(), -prod/2);
    // const auto nr_threads = 1;

    // ScalarTensor<q_val_t> plain_inputs{in, dim_t{in.size()}};
    // printf("plain inputs:\n");
    // plain_inputs.print();

    // const vector<crt_val_t> s {2,3,5,7};
    // std::cerr << "SCALING FACTOR IN MAIN: " << s[0] << std::endl;
    // auto circuit = new Circuit{new Rescale{s, plain_inputs.get_dims()}};
    // //auto circuit = new Circuit{new Sign{plain_inputs.get_dims()}};
    // auto plain_outputs = circuit->plain_q_eval(plain_inputs);
    // printf("plain output:\n");
    // plain_outputs.print();

    // vector<crt_val_t> crt_base{2, 3, 5, 7, 11, 13, 17};
    // vector<mrs_val_t> mrs_base{26, 3};

    // // auto gc = new GarbledCircuit(circuit, 9, (float)100.0);
    // auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);
    // // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(plain_inputs)};
    // // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // auto t1 = high_resolution_clock::now();
    // auto g_output{gc->cpu_evaluate(g_inputs, nr_threads)};
    // auto t2 = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(t2 - t1);
    // std::cerr << "BE evaluation time: "
    //       << duration.count() << " milliseconds, nr_threads:" << nr_threads << std::endl;
    // // gc->cuda_evaluate(g_dev_inputs);
    // // auto g_output{gc->cuda_move_outputs()};
    // auto outputs{gc->decode_outputs(g_output)};
    // printf("outputs:\n");
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    //
    //
    //Add BaseExtension Layer
    // const auto prod = 2*7;
    // vector<q_val_t> in(prod);
    // iota(in.begin(), in.end(), 0);
    // // const auto nr_threads = 16;

    // ScalarTensor<q_val_t> plain_inputs{in, dim_t{in.size()}};
    // printf("plain inputs:\n");
    // plain_inputs.print();

    // const vector<crt_val_t> scaling_factors {5,3};
    // auto circuit = new Circuit{new BaseExtension{plain_inputs.get_dims(), scaling_factors}};
    // auto plain_outputs = circuit->plain_q_eval(plain_inputs);
    // printf("plain output:\n");
    // plain_outputs.print();

    // vector<crt_val_t> crt_base{2, 3, 5, 7};
    // vector<mrs_val_t> mrs_base{26, 3};

    // // auto gc = new GarbledCircuit(circuit, 9, (float)100.0);
    // auto gc = new GarbledCircuit(circuit, crt_base, mrs_base);
    // // std::cerr << "base label for 5 in main():" << std::endl;
    // // gc->get_base_label().back()->print();
    // // std::cerr << "offset label for 5 in main():" << std::endl;
    // // gc->get_label_offset(5).print();

    // // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(plain_inputs)};
    // // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    // auto g_output{gc->cpu_evaluate(g_inputs)};
    // // gc->cuda_evaluate(g_dev_inputs);
    // // auto g_output{gc->cuda_move_outputs()};
    // auto outputs{gc->decode_outputs(g_output)};
    // printf("ungarbled outputs:\n");
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    //
    //
    //

    // //
    // //
    // // Example: Automatic inference of quantization constant
    // // auto dataset = cifar10("../../data/cifar10/cifar-10-batches-bin");
    // // vector<wandb_t> mean = {0.4914, 0.4822, 0.4465};
    // // vector<wandb_t> std = {0.247, 0.243, 0.261};
    // // normalize(&dataset, mean, std);

    // auto dataset = mnist("../../data/MNIST/raw");

    // // Create subset of dataset
    // vector<ScalarTensor<wandb_t>> small_test_images;
    // vector<unsigned long> small_test_labels;
    // for (int i = 0; i < 100; ++i) {
    //     auto tmp_image{dataset.testing_images.at(i)};
    //     auto tmp_label{dataset.testing_labels.at(i)};
    //     small_test_images.push_back(tmp_image);
    //     small_test_labels.push_back(tmp_label);
    // }

    // int target_crt_base_size = 7;

    // std::string model_path = "../../models/trained/MODEL_B.onnx";
    // Circuit* circuit =
    //     load_onnx_model(model_path, QuantizationMethod::SimpleQuant);
    // circuit->print();

    // int nr_samples = 10;
    // circuit->optimize_quantization(target_crt_base_size, dataset.testing_images,
    //                                0.1, 0.001, 0.0001, nr_samples);
    // auto q_const = circuit->get_q_const();

    // std::cout << "q_const: " << q_const << std::endl;

    // // auto quantized_inputs =
    // //     quantize(dataset.testing_images, QuantizationMethod::ScaleQuant, QL);
    // auto quantized_inputs = quantize(dataset.testing_images,
    // QuantizationMethod::SimpleQuant, q_const);

    // auto plain_test_acc =
    //     circuit->plain_test(dataset.testing_images, dataset.testing_labels);
    // printf("plain_test_acc: %f\n", plain_test_acc);

    // auto plain_q_test_acc =
    //     circuit->plain_q_test(quantized_inputs, dataset.testing_labels);
    // printf("plain_q_test_acc: %f\n", plain_q_test_acc);

    // // Needed to recompute q_range of all layers
    // // auto plain_outputs = circuit->plain_eval(small_test_images.at(0));
    // // std::cout << "plain outputs: " << std::endl;
    // // plain_outputs.print();

    // // auto plain_q_outputs = circuit->plain_q_eval(quantized_inputs);
    // // std::cout << "\nplain quantized outputs: " << std::endl;

    // // for (auto l : circuit->get_layer()) {
    // //     l->print();
    // //     auto min = l->get_min_plain_val();
    // //     auto max = l->get_max_plain_val();
    // //     auto range = l->get_range_plain_val();
    // //     std::cout << "min: " << min << std::endl;
    // //     std::cout << "max: " << max << std::endl;
    // //     std::cout << "range: " << range << std::endl;
    // //     auto q_min = l->get_min_plain_q_val();
    // //     auto q_max = l->get_max_plain_q_val();
    // //     auto q_range = l->get_range_plain_q_val();
    // //     std::cout << "q_min: " << q_min << std::endl;
    // //     std::cout << "q_max: " << q_max << std::endl;
    // //     std::cout << "q_range: " << q_range << std::endl;
    // //     std::cout << std::endl;
    // // }
    // // std::cout << "circuit min: " << circuit->get_min_plain_val() <<
    // // std::endl; std::cout << "circuit max: " << circuit->get_max_plain_val()
    // // << std::endl; std::cout << "circuit range: " <<
    // // circuit->get_range_plain_val()
    // //           << std::endl;
    // // std::cout << "circuit q_min: " << circuit->get_min_plain_q_val()
    // //           << std::endl;
    // // std::cout << "circuit q_max: " << circuit->get_max_plain_q_val()
    // //           << std::endl;
    // // std::cout << "circuit q_range: " << circuit->get_range_plain_q_val()
    // //           << std::endl;

    // auto gc = new GarbledCircuit(circuit, target_crt_base_size, (float)100.0);
    // gc->cuda_move();
    // auto g_inputs{gc->garble_inputs(quantized_inputs.at(0))};
    // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};

    // gc->cuda_evaluate(g_dev_inputs);
    // auto g_outputs{gc->cuda_move_outputs()};

    // auto outputs{gc->decode_outputs(g_outputs)};

    // std::cout << "output_label: " << outputs.argmax() << std::endl;

    // std::cout << "\noutputs: " << std::endl;
    // outputs.print();

    // // clean up
    // for (auto label : *g_inputs) {
    //     delete label;
    // }
    // delete g_inputs;
    // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // delete gc;
    // //
    // //
    // //

    //
    //
    // Example: Load and normalize CIFAR-10 dataset
    // auto dataset = cifar10("../../data/cifar10/cifar-10-batches-bin");
    // auto x = dataset.training_images.at(0);
    // x.print();
    // vector<wandb_t> mean = {0.4914, 0.4822, 0.4465};
    // vector<wandb_t> std = {0.247, 0.243, 0.261};
    // normalize(&dataset, mean, std);
    // x = dataset.training_images.at(0);
    // x.print();
    //
    //
    //

    //
    //
    //
    // std::mt19937 gen(42);
    // wandb_t q_const = 0.0001;

    // int input_width = 32;
    // int input_height = 32;
    // int input_size = input_height * input_width;
    // int channel = 3;

    // int filter = 64;
    // int filter_width = 3;
    // int filter_height = 3;
    // int filter_size = filter_width * filter_height;

    // int stride_width = 1;
    // int stride_height = 1;

    // dim_t weights_dims{filter_width, filter_height, channel, filter};
    // auto weights = init_w_conv2d(weights_dims, gen);
    // weights.print();
    // dim_t bias_dims{filter};
    // auto bias = init_b_conv2d(bias_dims, filter_size, channel, gen);
    // dim_t input_dims{input_width, input_height, channel};
    // auto inputs = init_inputs(input_dims, gen);

    // auto circuit = new Circuit{new Conv2d(
    //     weights, bias, input_width, input_height, channel, filter,
    //     filter_width, filter_height, stride_width, stride_height, q_const)};
    // auto acc = circuit->compute_q_acc(inputs, 1, q_const);
    // printf("acc: %f\n", acc);
    // int crt_base_size = circuit->infer_crt_base_size(inputs);
    // printf("crt_base_size: %d\n", crt_base_size);
    // // auto gc = new GarbledCircuit(circuit, crt_base_size);
    // // gc->cuda_move();
    // // auto g_inputs{gc->garble_inputs(inputs)};
    // // auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};

    // // auto g_outputs{gc->cuda_move_outputs()};
    // // auto outputs{gc->decode_outputs(g_outputs)};

    // // clean up
    // // for (auto label : *g_inputs) {
    // //     delete label;
    // // }
    // // delete g_inputs;
    // // gc->cuda_free_inputs(g_dev_inputs);

    // delete circuit;
    // // delete gc;
    //
    //
    //

    // Play with the LabelSlice Iterator
    // crt_val_t modulus = 19;
    // vector<crt_val_t> values(3*3*LabelTensor::get_nr_comps(modulus));
    // iota(values.begin(), values.end(), 0);
    // std::transform(values.begin(), values.end(), values.begin(),
    //                [modulus](crt_val_t val) { return val % modulus; });
    // LabelTensor tensor_a{modulus, values, dim_t{3, 3}};
    // LabelTensor tensor_b{modulus, values, dim_t{3, 3}};
    // tensor_a.print();

    // auto tensor_slice{tensor_a.get_label(4)};
    // // auto slice_a{tensor_a.slice({{0,0}, {0,2}})};
    // // auto slice_b{tensor_a.slice({{0,0}, {0,2}})};
    // tensor_b -= tensor_slice;

    // tensor_b.print();

}
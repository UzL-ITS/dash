#ifndef TEST_PROJECTION_H
#define TEST_PROJECTION_H

#include <vector>

#include "circuit/circuit.h"
#include "misc/datatypes.h"
#include "garbling/garbled_circuit.h"
#include "circuit/layer/projection.h"
#include "circuit/scalar_tensor.h"

using std::vector;

crt_val_t test_proj_function(crt_val_t x, void* params) { return x * 2; }
crt_val_t test_proj_function2(crt_val_t x, void* params) { return x / 2 - 10; }
crt_val_t test_proj_function3(crt_val_t x, void* params) { return x / 4; }

TEST(TestProjection, ProjectHigherModulusAndBack) {
    ScalarTensor<q_val_t> inputs{{15}, dim_t{1}};

    // in- and output moduli
    vector<crt_val_t> in_moduli{19};
    vector<crt_val_t> out_moduli{91};

    auto circuit = new Circuit{new Projection(inputs.get_dims(), in_moduli,
                                              out_moduli, &test_proj_function),
                               new Projection(inputs.get_dims(), out_moduli,
                                              in_moduli, &test_proj_function2)};

    vector<crt_val_t> crt_base{19};

    auto gc = new GarbledCircuit(circuit, crt_base, 100);
    auto g_inputs{gc->garble_inputs(inputs)};
    auto g_outputs{gc->cpu_evaluate(g_inputs)};

    auto outputs{gc->decode_outputs(g_outputs)};

    EXPECT_EQ(outputs.at(0), 5);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;

    delete circuit;
    delete gc;
}

TEST(TestProjection, ProjectLowerModulusAndBack) {
    ScalarTensor<q_val_t> inputs{{12}, dim_t{1}};

    // in- and output moduli
    vector<crt_val_t> in_moduli{19};
    vector<crt_val_t> out_moduli{10};

    auto circuit = new Circuit{new Projection(inputs.get_dims(), in_moduli,
                                              out_moduli, &test_proj_function3),
                               new Projection(inputs.get_dims(), out_moduli,
                                              in_moduli, &test_proj_function)};

    vector<crt_val_t> crt_base{19};

    auto gc = new GarbledCircuit(circuit, crt_base);
    auto g_inputs{gc->garble_inputs(inputs)};
    auto g_outputs{gc->cpu_evaluate(g_inputs)};

    auto outputs{gc->decode_outputs(g_outputs)};

    EXPECT_EQ(outputs.at(0), 6);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;

    delete circuit;
    delete gc;
}

TEST(TestProjection, GPUProjection) {
    ScalarTensor<q_val_t> inputs{{3, 2, 1, -1}, dim_t{2, 2}};

    // in- and output moduli
    vector<crt_val_t> in_moduli{19};
    vector<crt_val_t> out_moduli{19};

    auto circuit = new Circuit{new Projection(inputs.get_dims(), in_moduli,
                                              out_moduli, &test_proj_function)};

    vector<crt_val_t> crt_base{19};

    auto gc = new GarbledCircuit(circuit, crt_base, 100);
    gc->cuda_move();
    auto g_inputs{gc->garble_inputs(inputs)};
    auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{gc->cuda_move_outputs()};

    auto outputs{gc->decode_outputs(g_outputs)};

    vector<q_val_t> expected_outputs{6, 4, 2, -2};
    EXPECT_EQ(outputs.as_vector(), expected_outputs);
    EXPECT_EQ(outputs.get_dims(), inputs.get_dims());

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    gc->cuda_free_inputs(g_dev_inputs);

    delete circuit;
    delete gc;
}

TEST(TestProjection, ProjectHigherModulusAndBackGPU) {
    ScalarTensor<q_val_t> inputs{{15}, dim_t{1}};

    // in- and output moduli
    vector<crt_val_t> in_moduli{19};
    vector<crt_val_t> out_moduli{91};

    auto circuit = new Circuit{new Projection(inputs.get_dims(), in_moduli,
                                              out_moduli, &test_proj_function),
                               new Projection(inputs.get_dims(), out_moduli,
                                              in_moduli, &test_proj_function2)};

    vector<crt_val_t> crt_base{19};

    auto gc = new GarbledCircuit(circuit, crt_base, 100);
    gc->cuda_move();
    auto g_inputs{gc->garble_inputs(inputs)};
    auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{gc->cuda_move_outputs()};

    auto outputs{gc->decode_outputs(g_outputs)};

    EXPECT_EQ(outputs.at(0), 5);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    gc->cuda_free_inputs(g_dev_inputs);

    delete circuit;
    delete gc;
}

TEST(TestProjection, ProjectLowerModulusAndBackGPU) {
    ScalarTensor<q_val_t> inputs{{12}, dim_t{1}};

    // in- and output moduli
    vector<crt_val_t> in_moduli{19};
    vector<crt_val_t> out_moduli{10};

    auto circuit = new Circuit{new Projection(inputs.get_dims(), in_moduli,
                                              out_moduli, &test_proj_function3),
                               new Projection(inputs.get_dims(), out_moduli,
                                              in_moduli, &test_proj_function)};

    vector<crt_val_t> crt_base{19};

    auto gc = new GarbledCircuit(circuit, crt_base, 100);
    gc->cuda_move();
    auto g_inputs{gc->garble_inputs(inputs)};
    auto g_dev_inputs{gc->cuda_move_inputs(g_inputs)};
    gc->cuda_evaluate(g_dev_inputs);
    auto g_outputs{gc->cuda_move_outputs()};

    auto outputs{gc->decode_outputs(g_outputs)};

    EXPECT_EQ(outputs.at(0), 6);

    // clean up
    for (auto label : *g_inputs) {
        delete label;
    }
    delete g_inputs;
    gc->cuda_free_inputs(g_dev_inputs);

    delete circuit;
    delete gc;
}

#endif
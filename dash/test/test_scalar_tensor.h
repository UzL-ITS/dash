#ifndef TEST_SCALAR_TENSOR_H
#define TEST_SCALAR_TENSOR_H

#include "circuit/scalar_tensor.h"
#include "misc/datatypes.h"

using std::vector;

class TestScalarTensor : public ::testing::Test {};

TEST(TestScalarTensor, Quantize) {
    ScalarTensor<wandb_t> values{{-15.0, -14.0, 14.0, 15.0}, dim_t{4}};
    auto q_values = ScalarTensor<q_val_t>::quantize(
        values, QuantizationMethod::SimpleQuant, 10.0F);
    ScalarTensor<q_val_t> expected_values{{-2, -1, 1, 2}, dim_t{4}};
    EXPECT_EQ(q_values.as_vector(), expected_values.as_vector());
}

TEST(TestScalarTensor, Mod) {
    ScalarTensor<q_val_t> values{{0, 1, 2, 3, 4, -5, -4, -3, -2, -1},
                                 dim_t{10}};
    values.mod(10);
    ScalarTensor<q_val_t> expected_values{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                          dim_t{10}};
    EXPECT_EQ(values.as_vector(), expected_values.as_vector());
}

#endif
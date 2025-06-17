#include <gtest/gtest.h>

#include "misc/cuda_util.h"
#include "test_conv2d.h"
#include "test_cuda_aes_engine.h"
#include "test_cuda_util.h"
#include "test_dense.h"
#include "test_label.h"
#include "test_mixed_mod_mult.h"
#include "test_mult.h"
#include "test_onnx_modelloader.h"
#include "test_projection.h"
#include "test_relu.h"
#include "test_rescale.h"
#include "test_scalar_tensor.h"
#include "test_sign.h"
#include "test_util.h"
#include "test_maxpool2d.h"

int main(int argc, char **argv) {
    init_cuda();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
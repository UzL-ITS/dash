#ifndef DATATYPES_C_H
#define DATATYPES_C_H

#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <stdlib.h>
#include <map>
#include <vector>

#ifndef __SIZEOF_INT128__
#error "__uint128_t not defined!"
#endif

#define DEFAULT_NUM_THREADS 16

#define Q_VAL_MAX LLONG_MAX
#define Q_VAL_MIN LLONG_MIN

#define CRT_VAL_MAX SHRT_MAX
#define CRT_VAL_MIN SHRT_MIN

#define MRS_VAL_MAX SHRT_MAX
#define MRS_VAL_MIN SHRT_MIN

#define WANDB_VAL_MAX FLT_MAX
#define WANDB_VAL_MIN FLT_MIN

// TODO: this is only used in legacy Enclave code, deploy Circuit::get_l() there, too
// TODO: in linear layers, "5" is still often hardcoded
#define QL 5

// TODO: find better way to enable/disable sign-based legacy scaling
// #define USE_LEGACY_SCALING

typedef long long q_val_t;  // V
typedef int16_t crt_val_t;  // T
typedef int16_t mrs_val_t;  // U
typedef float wandb_t;  // do not change without modifieng the onnx model loader

// optimized CRT bases for different QL values, precomputed by the ReDASH authors
const std::map<int, std::vector<crt_val_t>> OPTIMAL_CRT_BASES = {
    {3, {3, 5, 7, 8, 11}},
    {4, {3, 5, 7, 11, 13, 16}},
    {5, {3, 5, 7, 11, 13, 32}}
    // Add more entries as needed
};

// optimized MRS bases for different QL values, precomputed by the ReDASH authors.
const std::map<int, std::vector<mrs_val_t>> OPTIMAL_MRS_BASES = {
    {3, {12, 6, 6, 6}},
    {4, {10, 8, 6, 5, 5, 5}},
    {5, {8, 8, 8, 7, 6, 6}}
    // Add more entries as needed
};

// scaling factors for different QL values in non-optimized CPM bases, precomputed by the ReDASH authors
// IMPORTANT: Assumes crt_base_size = 8!
const std::map<int, std::vector<int>> SCALING_FACTORS_CPM_BASES = {
    {3, {}},
    {4, {}},
    {5, {}}
    // Add more entries as needed
};

// used in read_file ocall in onnx_modelloader
typedef struct raw_data {
    char *data;
    size_t size;
} raw_data_t;
#endif
#ifndef DATATYPES_C_H
#define DATATYPES_C_H

#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <stdlib.h>

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

// \ell in the ReDASH paper
#define QL 5
// s in the ReDASH paper. Choose the member of the CRT base cloest to 2^QL.
#define QS 32

// optimal bases for different QL values, precomputed by the ReDASH authors
#if QL == 3
    #define OPTIMAL_CRT_BASE std::vector<crt_val_t>{3, 5, 7, 8, 11}
    #define OPTIMAL_MRS_BASE std::vector<mrs_val_t>{12, 6, 6, 6}
#elif QL == 4
    #define OPTIMAL_CRT_BASE std::vector<crt_val_t>{3, 5, 7, 11, 13, 16}
    #define OPTIMAL_MRS_BASE std::vector<mrs_val_t>{10, 8, 6, 5, 5, 5}
#elif QL == 5
    #define OPTIMAL_CRT_BASE std::vector<crt_val_t>{3, 5, 7, 11, 13, 32}
    #define OPTIMAL_MRS_BASE std::vector<mrs_val_t>{8, 8, 8, 7, 6, 6}
#else
    #error "Unsupported QL value: Cannot determine optimal bases."
#endif

typedef long long q_val_t;  // V
typedef int16_t crt_val_t;  // T
typedef int16_t mrs_val_t;  // U
typedef float wandb_t;  // do not change without modifieng the onnx model loader

// used in read_file ocall in onnx_modelloader
typedef struct raw_data {
    char *data;
    size_t size;
} raw_data_t;
#endif
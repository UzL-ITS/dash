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

//#define QL 8
#define QL 5

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
#ifndef MISC_H
#define MISC_H

#include "misc/datatypes.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

#ifndef SGX
#include <cuda_runtime_api.h>
#endif

#include <inttypes.h>

#include <cstdio>

#define P10_UINT64 10000000000000000000ULL
#define E10_UINT64 19

#define STRINGIZER(x) #x
#define TO_STRING(x) STRINGIZER(x)

/**
 * @brief      Print 128bit value.
 *
 * From https://stackoverflow.com/a/11660651/8538713.
 *
 * Produces warning in cuda code: ptxas warning : Stack size for entry function
 * '_Z9eval_multPsiPoS0_S_si' cannot be statically determined
 *
 * @param[in]  u128  Value.
 *
 * @return     Number of printed characters.
 */
#ifndef SGX
__host__ __device__
#endif
    int
    print_u128_u(__uint128_t u128) {
    int rc;
    if (u128 > UINT64_MAX) {
        __uint128_t leading = u128 / P10_UINT64;
        uint64_t trailing = u128 % P10_UINT64;
        rc = print_u128_u(leading);
#ifdef SGX
        rc += printf("%." TO_STRING(E10_UINT64) PRIu64, trailing);
#else
        rc += printf("%." TO_STRING(E10_UINT64) PRIu64, trailing);
#endif
    } else {
        uint64_t u64 = u128;
#ifdef SGX
        rc = printf("%" PRIu64, u64);
#else
        rc = printf("%" PRIu64, u64);
#endif
    }
    return rc;
}

#endif

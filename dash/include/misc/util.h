#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "circuit/scalar_tensor.h"
#include "misc/misc.h"

using std::vector;

// Ceil for integers
template <typename T>
inline T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

#ifdef SGX

inline void sgx_cudaMemcpyToDevice(void* dst, void* src, size_t size) {
    void* untrusted_buffer;
    ocall_alloc_array(&untrusted_buffer, size);
    std::memcpy(untrusted_buffer, src, size);
    ocall_cudaMemcpyToDevice(dst, untrusted_buffer, size);
}

#endif

namespace util {
/**
 * @brief Modulo operation.
 *
 * @tparam X
 * @param divisor
 * @param dividend
 * @return X
 */
template <typename X>
inline X modulo(X divisor, X dividend) {
    return (divisor % dividend + dividend) % dividend;
}

/**
 * @brief Modulo operation.
 *
 * @tparam V
 * @tparam T
 * @param divisor
 * @param dividend
 * @return T
 */
template <typename V, typename T>
inline T modulo(V divisor, T dividend) {
    V cast_dividend = static_cast<V>(dividend);
    V result = (divisor % cast_dividend + cast_dividend) % cast_dividend;
    return static_cast<T>(result);
}

template <typename T>
inline vector<T> sieve_of_eratosthenes(int nr_primes) {
    assert(nr_primes > 0 && "nr_primes must be positive");
    assert(nr_primes <= 100 && "nr_primes must be less than 100");
    vector<T> primes{2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,
                     41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,
                     97,  101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
                     157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
                     227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                     283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
                     367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
                     439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
                     509, 521, 523, 541};

    return vector<T>(primes.begin(), primes.begin() + nr_primes);
}

/**
 * @brief      Compute multiplicative inverse of a mod b.
 *
 * Code is from https://rosettacode.org/wiki/Chinese_remainder_theorem#C.
 *
 * @param[in]  a     Value.
 * @param[in]  b     Modulus.
 *
 * @return     Multiplicative inverse of value mod modulus.
 */
inline uint64_t mul_inv(__uint128_t a, uint16_t b) {
    long long b0 = b, t, q;
    long long x0 = 0, x1 = 1;
    if (b == 1) return 1;
    while (a > 1) {
        q = a / b;
        t = b, b = a % b, a = t;
        t = x0, x0 = x1 - q * x0, x1 = t;
    }
    if (x1 < 0) x1 += b0;
    return x1;
}

/**
 * @brief      Apply chinese remainder theorem to solve system of linear
 * 			   congruences.
 *
 * Code is from https://rosettacode.org/wiki/Chinese_remainder_theorem#C, but
 * modified.
 *
 * @param      moduli
 * @param      residues     Values.
 *
 * @return     Solution.
 */
template <typename T, typename V>
ScalarTensor<V> chinese_remainder(vector<T>& moduli,
                                  vector<ScalarTensor<T>>& residues) {
    ScalarTensor<V> result{};
    result.resize(residues.at(0).get_dims());

    uint64_t i;
    __uint128_t p;
    __uint128_t prod = std::accumulate(moduli.begin(), moduli.end(), 1,
                                       std::multiplies<__uint128_t>());

    for (size_t j = 0; j < residues.at(0).size(); ++j) {
        __uint128_t sum = 0;

        for (i = 0; i < moduli.size(); ++i) {
            p = prod / moduli.at(i);
            sum += ((residues.at(i).data()[j] *
                     (mul_inv(p, moduli.at(i)) % prod)) %
                    prod * p) %
                   prod;
        }
        result.data()[j] = static_cast<V>(sum % prod);
    }

    return result;
}

template <typename T, typename V>
vector<ScalarTensor<T>> crt_reduce(ScalarTensor<V>& values,
                                   vector<T>& crt_base) {
    vector<ScalarTensor<T>> reduced_values;
    reduced_values.resize(crt_base.size());
    for (size_t i = 0; i < crt_base.size(); ++i) {
        reduced_values.at(i) = ScalarTensor<T>(values.get_dims());
        for (size_t j = 0; j < values.size(); ++j) {
            reduced_values.at(i).push_back(
                modulo<V, T>(values.at(j), crt_base.at(i)));
        }
    }
    return reduced_values;
}

// For testing purposes.
template <typename T>
inline vector<T> get_random_vector(int size, T min, T max,
                                   unsigned int seed = 42) {
    static std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(min, max);
    vector<T> result;
    result.reserve(size);
    for (int i = 0; i < size; i++) {
        result.push_back(dis(gen));
    }
    return result;
}

}  // namespace util

#endif
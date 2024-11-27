#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <list>
#include <vector>

#include "misc/datatypes.h"
#include "circuit/scalar_tensor.h"
#include "misc/util.h"

using std::vector;

template <typename X>
class TestUtilSigned : public ::testing::Test {};
using signed_types = ::testing::Types<int, long, long long>;
TYPED_TEST_SUITE(TestUtilSigned, signed_types);

TYPED_TEST(TestUtilSigned, SignedModulo) {
    EXPECT_EQ(90, util::modulo(-10, 100));
    EXPECT_EQ(-90, util::modulo(10, -100));
    EXPECT_EQ(-10, util::modulo(-10, -100));
}

template <typename X>
class TestUtil : public ::testing::Test {};
using unsigned_types = ::testing::Types<uint8_t, uint16_t>;
TYPED_TEST_SUITE(TestUtil, unsigned_types);

TYPED_TEST(TestUtil, UnsignedModulo) {
    EXPECT_EQ(0, util::modulo(10, 10));
    EXPECT_EQ(10, util::modulo(10, 100));
}

TEST(TestUtil, ChineseRemainder) {
    vector<crt_val_t> residues{0, 1, 0, 4, 1, 3, 4, 6};
    vector<crt_val_t> crt_base{2, 3, 5, 7, 11, 13, 17, 19};

    vector<ScalarTensor<crt_val_t>> residue_tensors{};
    for (auto &val : residues) {
        residue_tensors.push_back(ScalarTensor<crt_val_t>{{val}, dim_t{1}});
    }

    auto value =
        util::chinese_remainder<crt_val_t, q_val_t>(crt_base, residue_tensors);
    EXPECT_EQ(value.data()[0], 10000);
}

TEST(TestUtil, SieveOfEratosthenes) {
    vector<crt_val_t> primes = util::sieve_of_eratosthenes<crt_val_t>(15);
    vector<crt_val_t> expected_primes{2,  3,  5,  7,  11, 13, 17, 19,
                                      23, 29, 31, 37, 41, 43, 47};
    EXPECT_EQ(primes, expected_primes);
}

#endif
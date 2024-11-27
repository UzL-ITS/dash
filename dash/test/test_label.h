#ifndef TEST_LABEL_H
#define TEST_LABEL_H

#include <cstdint>
#include <numeric>
#include <vector>

#include "garbling/label_tensor.h"
#include "misc/util.h"

using std::vector;

class TestLabel : public ::testing::Test {};

template <typename X>
inline X add_mod(X a, X b, X modulus) {
    return util::modulo<crt_val_t>(a + b, modulus);
}

template <typename X>
inline X sub_mod(X a, X b, X modulus) {
    return util::modulo<crt_val_t>(a - b, modulus);
}

// Basic Operations
//// LT means LabelTensor
//// TS means TensorSlice
//// ST means ScalarTensor
//// OP_LTLT means operation between two LabelTensors

TEST(TestLabel, CompoundAddLTLT) {
    crt_val_t modulus{7};
    dim_t dims{2, 2};
    size_t nr_comps{LabelTensor::get_nr_comps(modulus)};

    vector<crt_val_t> a_vec(2 * 2 * nr_comps);
    std::iota(a_vec.begin(), a_vec.end(), 0);
    std::transform(a_vec.begin(), a_vec.end(), a_vec.begin(),
                   [modulus](crt_val_t x) { return x % modulus; });
    vector<crt_val_t> b_vec(2 * 2 * nr_comps);
    std::iota(b_vec.begin(), b_vec.end(), 0);
    std::transform(b_vec.begin(), b_vec.end(), b_vec.begin(),
                   [modulus](crt_val_t x) { return x % modulus; });
    vector<crt_val_t> c_vec(2 * 2 * nr_comps);
    std::transform(a_vec.begin(), a_vec.end(), b_vec.begin(), c_vec.begin(),
                   [modulus](crt_val_t x, crt_val_t y) {
                       return add_mod<crt_val_t>(x, y, modulus);
                   });

    LabelTensor a_l{modulus, a_vec, dims};
    LabelTensor b_l{modulus, b_vec, dims};
    a_l += b_l;

    EXPECT_EQ(a_l.as_vector(), c_vec);
}

TEST(TestLabel, CompoundAddLTTS) {
    crt_val_t modulus{7};
    dim_t dim_a{2, 2};
    dim_t dim_b{3, 3};
    size_t nr_comps{LabelTensor::get_nr_comps(modulus)};

    vector<crt_val_t> a_vec(2 * 2 * nr_comps);
    std::iota(a_vec.begin(), a_vec.end(), 0);
    std::transform(a_vec.begin(), a_vec.end(), a_vec.begin(),
                   [modulus](crt_val_t x) { return x % modulus; });
    vector<crt_val_t> b_vec(3 * 3 * nr_comps);
    std::iota(b_vec.begin(), b_vec.end(), 0);
    std::transform(b_vec.begin(), b_vec.end(), b_vec.begin(),
                   [modulus](crt_val_t x) { return x % modulus; });
    vector<crt_val_t> c_vec(2 * 2 * nr_comps);
    for (size_t i = 0; i < 2 * 2; ++i) {
        size_t j = (i % 2) + (i / 2) * 3;
        for (size_t k = 0; k < nr_comps; ++k) {
            c_vec[i * nr_comps + k] = add_mod<crt_val_t>(
                a_vec[i * nr_comps + k], b_vec[j * nr_comps + k], modulus);
        }
    }

    LabelTensor a_l{modulus, a_vec, dim_a};
    LabelTensor b_l{modulus, b_vec, dim_b};
    auto b_s{b_l.slice({{0, 1}, {0, 1}})};
    a_l += b_s;

    EXPECT_EQ(a_l.as_vector(), c_vec);
}

TEST(TestLabel, CompoundSubLTLT) {
    crt_val_t modulo{7};
    vector<crt_val_t> a(LabelTensor::get_nr_comps(modulo));
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(),
                   [modulo](crt_val_t x) { return x % modulo; });
    vector<crt_val_t> b(LabelTensor::get_nr_comps(modulo));
    std::iota(b.begin(), b.end(), 1);
    std::transform(b.begin(), b.end(), b.begin(),
                   [modulo](crt_val_t x) { return x % modulo; });

    LabelTensor a_l{modulo, a};
    LabelTensor b_l{modulo, b};

    std::transform(a.begin(), a.end(), b.begin(), a.begin(),
                   [modulo](crt_val_t x, crt_val_t y) {
                       return sub_mod<crt_val_t>(x, y, modulo);
                   });

    a_l -= b_l;
    EXPECT_EQ(a, a_l.get_components_vec());
}

TEST(TestLabel, CompoundSubLTTS) {
    crt_val_t modulus{7};
    dim_t dim_a{2, 2};
    dim_t dim_b{3, 3};
    size_t nr_comps{LabelTensor::get_nr_comps(modulus)};

    vector<crt_val_t> a_vec(2 * 2 * nr_comps);
    std::iota(a_vec.begin(), a_vec.end(), 0);
    std::transform(a_vec.begin(), a_vec.end(), a_vec.begin(),
                   [modulus](crt_val_t x) { return x % modulus; });
    vector<crt_val_t> b_vec(3 * 3 * nr_comps);
    std::iota(b_vec.begin(), b_vec.end(), 0);
    std::transform(b_vec.begin(), b_vec.end(), b_vec.begin(),
                   [modulus](crt_val_t x) { return x % modulus; });
    vector<crt_val_t> c_vec(2 * 2 * nr_comps);
    for (size_t i = 0; i < 2 * 2; ++i) {
        size_t j = (i % 2) + (i / 2) * 3;
        for (size_t k = 0; k < nr_comps; ++k) {
            c_vec[i * nr_comps + k] = sub_mod<crt_val_t>(
                a_vec[i * nr_comps + k], b_vec[j * nr_comps + k], modulus);
        }
    }

    LabelTensor a_l{modulus, a_vec, dim_a};
    LabelTensor b_l{modulus, b_vec, dim_b};
    auto b_s{b_l.slice({{0, 1}, {0, 1}})};
    a_l -= b_s;

    EXPECT_EQ(a_l.as_vector(), c_vec);
}

TEST(TestLabel, CompoundMultLT) {
    crt_val_t modulo{7};
    vector<crt_val_t> a(LabelTensor::get_nr_comps(modulo));
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(),
                   [modulo](crt_val_t x) { return x % modulo; });
    vector<crt_val_t> b(LabelTensor::get_nr_comps(modulo));

    LabelTensor a_l{modulo, a};

    std::transform(a.begin(), a.end(), b.begin(),
                   [modulo](crt_val_t x) { return (x * 10) % modulo; });

    a_l *= (crt_val_t)10;
    EXPECT_EQ(b, a_l.get_components_vec());
}

TEST(TestLabel, CompoundMultLTST) {
    crt_val_t modulo{7};
    size_t label_length = LabelTensor::get_nr_comps(modulo);
    size_t nr_labels = 3;

    vector<crt_val_t> a(nr_labels * label_length);
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(),
                   [modulo](crt_val_t x) { return x % modulo; });

    vector<q_val_t> b(nr_labels);
    std::iota(b.begin(), b.end(), 0);
    ScalarTensor<q_val_t> b_t{b, dim_t{nr_labels}};

    LabelTensor a_l{modulo, a, dim_t{nr_labels}};

    for (int i = 0; i < nr_labels; ++i) {
        for (int j = 0; j < label_length; ++j) {
            a[i * label_length + j] = (a[i * label_length + j] * b[i]) % modulo;
        }
    }

    a_l *= b_t;

    EXPECT_EQ(a, a_l.as_vector());
}

// Label Operations
TEST(TestLabel, CRTReduce) {
    ScalarTensor<q_val_t> values = {{7, 2, 4}, dim_t{3}};
    vector<crt_val_t> crt_modulus = {2, 3, 5, 7};
    auto reduced_values = util::crt_reduce<>(values, crt_modulus);
    vector<ScalarTensor<crt_val_t>> expected_values = {{{1, 0, 0}, dim_t{3}},
                                                       {{1, 2, 1}, dim_t{3}},
                                                       {{2, 2, 4}, dim_t{3}},
                                                       {{0, 2, 4}, dim_t{3}}};

    for (int i = 0; i < reduced_values.size(); ++i)
        EXPECT_EQ(reduced_values.at(i).as_vector(),
                  expected_values.at(i).as_vector());
}

// Check wether compress and decompress are inverse operations.
TEST(TestLabel, Compression) {
    crt_val_t modulo{19};
    vector<crt_val_t> a(LabelTensor::get_nr_comps(modulo));
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(),
                   [modulo](crt_val_t x) { return x % modulo; });

    LabelTensor a_l{modulo, a};
    a_l.compress();
    a_l.get_components()[0] = 42;
    EXPECT_NE(a, a_l.get_components_vec());
    a_l.decompress();
    EXPECT_EQ(a, a_l.get_components_vec());
}

TEST(TestLabel, FastDecompression) {
    crt_val_t modulo{7};
    vector<crt_val_t> a(LabelTensor::get_nr_comps(modulo));
    std::iota(a.begin(), a.end(), 0);
    std::transform(a.begin(), a.end(), a.begin(),
                   [modulo](crt_val_t x) { return x % modulo; });

    LabelTensor a_l{modulo, a};
    a_l.compress();
    a_l.get_components()[0] = 42;
    EXPECT_NE(a, a_l.get_components_vec());
    a_l.fast_decompress();
    EXPECT_EQ(a, a_l.get_components_vec());
}

class TestLabelValues : public ::testing::TestWithParam<crt_val_t> {
   protected:
    LabelTensor m_l_min{GetParam(), 0};
    LabelTensor m_l_max{GetParam(), static_cast<crt_val_t>(GetParam() - 1)};
};

TEST_P(TestLabelValues, FastDecompression) {
    m_l_min.compress();
    memset(m_l_min.get_components(), 1, m_l_min.get_nr_comps());
    m_l_min.fast_decompress();
    vector<crt_val_t> expected_min(m_l_min.get_nr_comps(), 0);
    EXPECT_EQ(m_l_min.get_components_vec(), expected_min);

    m_l_max.compress();
    memset(m_l_max.get_components(), 1, m_l_max.get_nr_comps());
    m_l_max.fast_decompress();
    vector<crt_val_t> expected_max(m_l_max.get_nr_comps(),
                                   static_cast<crt_val_t>(GetParam() - 1));
    EXPECT_EQ(m_l_max.get_components_vec(), expected_max);
}

TEST_P(TestLabelValues, Decompression) {
    m_l_min.compress();
    memset(m_l_min.get_components(), 1, m_l_min.get_nr_comps());
    m_l_min.decompress();
    vector<crt_val_t> expected_min(m_l_min.get_nr_comps(), 0);
    EXPECT_EQ(m_l_min.get_components_vec(), expected_min);

    m_l_max.compress();
    memset(m_l_max.get_components(), 1, m_l_max.get_nr_comps());

    m_l_max.decompress();
    vector<crt_val_t> expected_max(m_l_max.get_nr_comps(),
                                   static_cast<crt_val_t>(GetParam() - 1));
    EXPECT_EQ(m_l_max.get_components_vec(), expected_max);
}

INSTANTIATE_TEST_SUITE_P(
    TestLabel, TestLabelValues,
    ::testing::ValuesIn(util::sieve_of_eratosthenes<crt_val_t>(11)));

#endif
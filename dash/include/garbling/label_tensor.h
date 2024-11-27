#ifndef LABEL_TENSOR_H
#define LABEL_TENSOR_H

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "circuit/scalar_tensor.h"
#include "crypto/cpu_aes_engine.h"
#include "misc/datatypes.h"
#include "misc/misc.h"
#include "misc/util.h"

#ifdef SGX
#include "misc/enclave_functions.h"
#else
#include "crypto/cuda_aes_engine.h"
#endif

#define MAX_LABEL_MODULO 31
#define CHUNK_LENGTH 16
#define NR_CHUNKS 8

using std::vector;
using slice_dim_t = std::vector<std::pair<size_t, size_t>>;

class LabelTensor;
template <typename LabelSlice, bool Const>
class LabelSliceIterator;

class LabelSlice {
    LabelTensor& m_tensor;
    vector<size_t> m_label_ids;
    slice_dim_t m_slice_dims;
    size_t m_slice_size;  // nr of labels in this slice

   public:
    using ValueType = crt_val_t;
    using Iterator = LabelSliceIterator<LabelSlice, false>;
    using ConstIterator = LabelSliceIterator<LabelSlice, true>;

   public:
    LabelSlice(LabelTensor& tensor, size_t label_idx);
    LabelSlice(LabelTensor& tensor, slice_dim_t& slice);

    LabelTensor& get_tensor() const;
    crt_val_t get_modulus() const;
    size_t get_nr_comps() const;
    crt_val_t* get_components() const;
    __uint128_t* get_compressed() const;
    __uint128_t* get_hashed() const;
    crt_val_t get_color() const;
    size_t get_slice_size() const;
    slice_dim_t get_slice_dims() const;
    void print() const;
    void operator+=(const LabelSlice&);
    void operator-=(const LabelSlice&);
    void operator*=(const q_val_t);
    friend LabelTensor operator*(const LabelSlice&, const q_val_t);
    friend LabelTensor operator*(const q_val_t, const LabelSlice&);
    friend LabelTensor operator+(const LabelSlice&, const LabelSlice&);
    friend LabelTensor operator-(const LabelSlice&, const LabelSlice&);
    Iterator begin();
    Iterator end();
    ConstIterator cbegin() const;
    ConstIterator cend() const;
};

template <typename LabelSlice, bool Const>
class LabelSliceIterator {
   private:
    std::conditional_t<Const, const LabelSlice&, LabelSlice&> m_slice;
    slice_dim_t m_slice_dims;
    vector<size_t> m_slice_idx;
    crt_val_t* m_value_ptr;

   public:
    LabelSliceIterator(
        std::conditional_t<Const, const LabelSlice&, LabelSlice&> slice)
        : m_slice(slice), m_slice_dims(slice.get_slice_dims()) {
        m_slice_idx.reserve(m_slice.get_slice_size());
        for (auto& p : m_slice_dims) m_slice_idx.push_back(p.first);
        m_slice_dims.back().second++;
        m_value_ptr = m_slice.get_tensor().at(m_slice_idx);
    }

    LabelSliceIterator(
        std::conditional_t<Const, const LabelSlice&, LabelSlice&> slice,
        vector<size_t> slice_idx)
        : m_slice(slice),
          m_slice_dims(slice.get_slice_dims()),
          m_slice_idx(slice_idx) {
        m_slice_dims.back().second++;
        m_value_ptr = m_slice.get_tensor().at(slice_idx);
    }

    LabelSliceIterator& operator++() {
        for (size_t i = 0; i < m_slice_idx.size(); ++i) {
            if (m_slice_idx.at(i) < m_slice_dims.at(i).second) {
                ++m_slice_idx.at(i);
                break;
            } else {
                m_slice_idx.at(i) = m_slice_dims.at(i).first;
            }
        }
        m_value_ptr = m_slice.get_tensor().at(m_slice_idx);
        return *this;
    }

    LabelSliceIterator& operator++(int) {
        LabelSliceIterator iterator = *this;
        ++(*this);
        return iterator;
    }

    LabelSliceIterator& operator--() {
        for (size_t i = 0; i < m_slice_idx.size(); ++i) {
            if (m_slice_idx.at(i) > m_slice_dims.at(i).first) {
                --m_slice_idx.at(i);
                break;
            } else {
                m_slice_idx.at(i) = m_slice_dims.at(i).second;
            }
        }
        m_value_ptr = m_slice.get_tensor().at(m_slice_idx);
        return *this;
    }

    LabelSliceIterator& operator--(int) {
        LabelSliceIterator iterator = *this;
        --(*this);
        return iterator;
    }

    std::conditional_t<Const, const crt_val_t&, crt_val_t&> operator[](
        size_t index) const {
        return m_value_ptr[index];
    }

    std::conditional_t<Const, const crt_val_t*, crt_val_t*> operator*() const {
        return m_value_ptr;
    }

    bool operator==(const LabelSliceIterator& other) const {
        return std::equal(m_slice_idx.begin(), m_slice_idx.end(),
                          other.m_slice_idx.begin());
    }

    bool operator!=(const LabelSliceIterator& other) const {
        return !(std::equal(m_slice_idx.begin(), m_slice_idx.end(),
                            other.m_slice_idx.begin()));
    }

    size_t get_label_idx() const {
        size_t idx{0};
        size_t offset{1};
        for (size_t i = 0; i < m_slice_idx.size(); ++i) {
            idx += m_slice_idx.at(i) * offset;
            offset *= m_slice.get_tensor().get_dims().at(i);
        }
        return idx;
    }
};

template <typename LabelTensor, bool Const>
class LabelTensorIterator {
   private:
    crt_val_t* m_value_ptr;
    std::conditional_t<Const, const LabelTensor&, LabelTensor&> m_tensor;

   public:
    LabelTensorIterator(
        std::conditional_t<Const, const LabelTensor&, LabelTensor&> tensor)
        : m_tensor(tensor) {
        m_value_ptr = m_tensor.get_components();
    }

    LabelTensorIterator(
        std::conditional_t<Const, const LabelTensor&, LabelTensor&> tensor,
        crt_val_t* value_ptr)
        : m_tensor(tensor), m_value_ptr(value_ptr) {}

    LabelTensorIterator& operator++() {
        m_value_ptr += m_tensor.get_nr_comps();
        return *this;
    }

    LabelTensorIterator& operator++(int) {
        LabelTensorIterator iterator = *this;
        ++(*this);
        return iterator;
    }

    LabelTensorIterator& operator--() {
        m_value_ptr -= m_tensor.get_nr_comps();
        return *this;
    }

    LabelTensorIterator& operator--(int) {
        LabelTensorIterator iterator = *this;
        --(*this);
        return iterator;
    }

    std::conditional_t<Const, const crt_val_t&, crt_val_t&> operator[](
        size_t index) const {
        return m_value_ptr[index];
    }

    std::conditional_t<Const, const crt_val_t*, crt_val_t*> operator*() const {
        return m_value_ptr;
    }

    bool operator==(const LabelTensorIterator& other) const {
        return m_value_ptr == other.m_value_ptr;
    }

    bool operator!=(const LabelTensorIterator& other) const {
        return m_value_ptr != other.m_value_ptr;
    }
};

class LabelTensor {
    crt_val_t m_modulus;
    size_t m_nr_comps;  // nr of components per label
    dim_t m_dims;
    crt_val_t* m_components{nullptr};
    size_t m_size;  // nr of all components over all label
    size_t m_nr_label;
    __uint128_t* m_compressed{nullptr};
    __uint128_t* m_hashed{nullptr};

   public:
    // used in- and outside of sgx
    // - static inline members are not supported in <c++17
    static inline CPUAESEngine m_cpu_aes_engine{};
    // The cuda AES engine gets initialized in the cuda_move method of the
    // garbled circuit class

    static inline const uint16_t* get_decompress_lookup(int modulus) {
        static const auto decompress_lookup =
            vector<vector<uint16_t>>{gen_decompress_lookup()};
        return decompress_lookup.at(modulus).data();
    }

    LabelTensor() = default;

    /**
     * @brief Construct label with given modulus and random components
     *
     * @param modulus
     */
    LabelTensor(crt_val_t modulus, dim_t dims = dim_t{1})
        : m_modulus{modulus},
          m_nr_comps{get_nr_comps(m_modulus)},
          m_dims{dims} {
        assert(m_nr_comps > 0 && "LabelTensor needs at least one component");
        m_nr_label = std::accumulate(dims.begin(), dims.end(), 1lu,
                                     std::multiplies<size_t>());
        m_size = m_nr_label * m_nr_comps;
        m_components = new crt_val_t[m_size];
    }

    /**
     * @brief Construct label with given modulus and single value
     *
     * @param modulus
     * @param value
     */
    LabelTensor(crt_val_t modulus, crt_val_t value, dim_t dims = dim_t{1})
        : m_modulus{modulus},
          m_dims{dims},
          m_nr_comps{get_nr_comps(m_modulus)} {
        assert(m_nr_comps > 0 && "LabelTensor needs at least one component");
        m_nr_label = std::accumulate(dims.begin(), dims.end(), 1lu,
                                     std::multiplies<size_t>());
        m_size = m_nr_label * m_nr_comps;
        m_components = new crt_val_t[m_size];
        std::fill(m_components, m_components + m_size, value);
    }

    /**
     * @brief Construct label with given modulus and values from C-style array
     *
     * @param modulus
     * @param values
     * @param nr_comps
     */
    LabelTensor(crt_val_t modulus, crt_val_t* values, size_t nr_comps,
                dim_t dims = dim_t{1})
        : m_modulus{modulus}, m_nr_comps{nr_comps}, m_dims{dims} {
        assert(m_nr_comps > 0 && "LabelTensor needs at least one component");
        m_nr_label = std::accumulate(dims.begin(), dims.end(), 1lu,
                                     std::multiplies<size_t>());
        m_size = m_nr_label * m_nr_comps;
        m_components = new crt_val_t[m_size];

        std::memcpy(m_components, values, m_size * sizeof(crt_val_t));
    }

    /**
     * @brief Construct label from given modulus and components from vector
     *
     * @param components
     * @param modulus
     */
    LabelTensor(crt_val_t modulus, vector<crt_val_t>& components,
                dim_t dims = dim_t{1})
        : m_modulus{modulus},
          m_nr_comps{get_nr_comps(m_modulus)},
          m_dims{dims} {
        m_nr_label = std::accumulate(dims.begin(), dims.end(), 1lu,
                                     std::multiplies<size_t>());
        m_size = m_nr_label * m_nr_comps;
        m_components = new crt_val_t[m_size];

        assert(components.size() == m_size && "LabelTensor-length mismatch");
        std::memcpy(m_components, components.data(),
                    m_size * sizeof(crt_val_t));
    }

    /**
     * @brief Copy constructor (from LabelTensor)
     *
     * @param label
     */
    LabelTensor(const LabelTensor& label)
        : m_modulus(label.m_modulus),
          m_nr_comps(label.m_nr_comps),
          m_dims(label.m_dims),
          m_size(label.m_size),
          m_nr_label(label.m_nr_label),
          m_compressed{nullptr},
          m_hashed{nullptr} {
        assert(m_nr_comps > 0 && "LabelTensor needs at least one component");

        // Copy label components
        m_components = new crt_val_t[m_size];
        std::memcpy(m_components, label.m_components,
                    m_size * sizeof(crt_val_t));

        // Copy compressed label representation
        if (label.m_compressed != nullptr) {
            m_compressed = new __uint128_t[m_nr_label];
            std::memcpy(m_compressed, label.m_compressed,
                        m_nr_label * sizeof(__uint128_t));
        }

        // Copy encrypted label representation
        if (label.m_hashed != nullptr) {
            m_hashed = new __uint128_t[m_nr_label];
            std::memcpy(m_hashed, label.m_hashed,
                        m_nr_label * sizeof(__uint128_t));
        }
    }

    /**
     * @brief Copy constructor (from LabelSlice)
     *
     * @param label
     */
    LabelTensor(const LabelSlice& view)
        : m_modulus(view.get_modulus()),
          m_nr_comps(view.get_nr_comps()),
          m_dims(dim_t{1}),
          m_size(view.get_nr_comps()),
          m_nr_label(1),
          m_compressed{nullptr},
          m_hashed{nullptr} {
        // Copy label components
        m_components = new crt_val_t[m_size];
        std::memcpy(m_components, view.get_components(),
                    m_size * sizeof(crt_val_t));

        // Copy compressed label representation
        if (view.get_compressed() != nullptr) {
            m_compressed = new __uint128_t[m_nr_label];
            std::memcpy(m_compressed, view.get_compressed(),
                        m_nr_label * sizeof(__uint128_t));
        }

        // Copy encrypted label representation
        if (view.get_hashed() != nullptr) {
            m_hashed = new __uint128_t[m_nr_label];
            std::memcpy(m_hashed, view.get_hashed(),
                        m_nr_label * sizeof(__uint128_t));
        }
    }

    /**
     * @brief Copy constructor with dimension extension (from LabelTensor)
     *
     * @param label
     */
    LabelTensor(const LabelTensor& label, dim_t dims)
        : m_modulus(label.m_modulus),
          m_nr_comps(label.m_nr_comps),
          m_dims(dims) {
        m_nr_label = std::accumulate(dims.begin(), dims.end(), 1lu,
                                     std::multiplies<size_t>());
        m_size = m_nr_label * m_nr_comps;
        m_components = new crt_val_t[m_size];

        assert(m_nr_label % label.m_nr_label == 0 &&
               "LabelTensor-length mismatch");

        // Copy label components
        for (size_t i = 0; i < m_nr_label / label.m_nr_label; ++i) {
            std::memcpy(m_components + i * m_nr_comps * label.m_nr_label,
                        label.m_components,
                        m_nr_comps * label.m_nr_label * sizeof(crt_val_t));
        }

        // Copy compressed label representation
        if (label.m_compressed != nullptr) {
            for (size_t i = 0; i < m_nr_label / label.m_nr_label; ++i) {
                std::memcpy(m_compressed + i * label.m_nr_label,
                            label.m_compressed,
                            label.m_nr_label * sizeof(__uint128_t));
            }
        }

        // Copy encrypted label representation
        if (label.m_hashed != nullptr) {
            for (size_t i = 0; i < m_nr_label / label.m_nr_label; ++i) {
                std::memcpy(m_hashed + i * label.m_nr_label, label.m_hashed,
                            label.m_nr_label * sizeof(__uint128_t));
            }
        }
    }

    /**
     * @brief Copy constructor with dimension extension (from LabelSlice)
     *
     * @param label
     */
    LabelTensor(const LabelSlice& view, dim_t dims)
        : m_modulus(view.get_modulus()),
          m_nr_comps(view.get_nr_comps()),
          m_dims(dims) {
        m_nr_label = std::accumulate(dims.begin(), dims.end(), 1lu,
                                     std::multiplies<size_t>());
        m_size = m_nr_label * m_nr_comps;
        m_components = new crt_val_t[m_size];

        // Copy label components
        for (size_t i = 0; i < m_nr_label; ++i) {
            std::memcpy(m_components + i * m_nr_comps, view.get_components(),
                        m_nr_comps * sizeof(crt_val_t));
        }

        // Copy compressed label representation
        if (view.get_compressed() != nullptr) {
            for (size_t i = 0; i < m_nr_label; ++i) {
                std::memcpy(m_compressed + i, view.get_compressed(),
                            sizeof(__uint128_t));
            }
        }

        // Copy encrypted label representation
        if (view.get_hashed() != nullptr) {
            for (size_t i = 0; i < m_nr_label; ++i) {
                std::memcpy(m_hashed + i, view.get_hashed(),
                            sizeof(__uint128_t));
            }
        }
    }

    virtual ~LabelTensor() {
        delete[] m_components;
        if (m_compressed != nullptr) delete[] m_compressed;
        if (m_hashed != nullptr) delete[] m_hashed;
    }

   public:
    // Label access
    crt_val_t* at(std::initializer_list<size_t> indices) {
        assert(indices.size() <= m_dims.size() && "Wrong number of indices");
        const size_t* idxs{indices.begin()};
        crt_val_t* ptr{m_components + (idxs[0] * m_nr_comps)};

        size_t offset = m_dims[0] * m_nr_comps;
        for (size_t i = 1; i < indices.size(); ++i) {
            ptr += idxs[i] * offset;
            offset *= m_dims[i];
        }
        return ptr;
    }

    crt_val_t* at(vector<size_t> indices) {
        assert(indices.size() <= m_dims.size() && "Wrong number of indices");
        size_t* idxs{&indices.front()};
        crt_val_t* ptr{m_components + (idxs[0] * m_nr_comps)};

        size_t offset = m_dims[0] * m_nr_comps;
        for (size_t i = 1; i < indices.size(); ++i) {
            ptr += idxs[i] * offset;
            offset *= m_dims[i];
        }
        return ptr;
    }

    // Basic operations
    LabelTensor& operator=(const LabelTensor& l) {
        m_modulus = l.m_modulus;
        m_nr_comps = l.m_nr_comps;
        m_nr_label = l.m_nr_label;
        m_size = l.m_size;
        m_dims = l.m_dims;
        if (m_components != nullptr) delete[] m_components;
        m_components = new crt_val_t[m_size];
        std::memcpy(m_components, l.m_components, m_size * sizeof(crt_val_t));
        if (l.m_compressed != nullptr) {
            delete[] m_compressed;
            m_compressed = new __uint128_t[m_nr_label];
            std::memcpy(m_compressed, l.m_compressed,
                        m_nr_label * sizeof(__uint128_t));
        }
        if (l.m_hashed != nullptr) {
            delete[] m_hashed;
            m_hashed = new __uint128_t[m_nr_label];
            std::memcpy(m_hashed, l.m_hashed, m_nr_label * sizeof(__uint128_t));
        }
        return *this;
    }

    LabelTensor& operator=(const LabelSlice& s) {
        m_modulus = s.get_modulus();
        m_nr_comps = s.get_nr_comps();
        m_nr_label = 1;
        m_size = m_nr_comps;
        m_dims = dim_t{1};
        if (m_components != nullptr) delete[] m_components;
        m_components = new crt_val_t[m_size];
        for (auto val = s.cbegin(); val != s.cend(); ++val) {
            std::memcpy(m_components, *val, m_size * sizeof(crt_val_t));
        }
        if (s.get_compressed() != nullptr) {
            delete[] m_compressed;
            m_compressed = new __uint128_t[m_nr_label];
            size_t idx = 0;
            for (auto val = s.cbegin(); val != s.cend(); ++val) {
                auto label_idx = val.get_label_idx();
                m_compressed[idx] = s.get_compressed()[label_idx];
                ++idx;
            }
        }
        if (s.get_hashed() != nullptr) {
            delete[] m_hashed;
            m_hashed = new __uint128_t[m_nr_label];
            size_t idx = 0;
            for (auto val = s.cbegin(); val != s.cend(); ++val) {
                auto label_idx = val.get_label_idx();
                m_hashed[idx] = s.get_hashed()[label_idx];
                ++idx;
            }
        }

        return *this;
    }

    void operator+=(const LabelTensor& l) {
        assert((m_nr_label == l.m_nr_label || l.m_nr_label == 1) &&
               "LabelTensor size missmatch");

        if (m_size == l.m_size) {
            for (size_t idx = 0; idx < m_nr_label; ++idx) {
                for (size_t i = 0; i < m_nr_comps; ++i) {
                    m_components[idx * m_nr_comps + i] =
                        util::modulo<crt_val_t>(
                            (m_components[idx * m_nr_comps + i] +
                             l.m_components[idx * m_nr_comps + i]),
                            m_modulus);
                }
            }
        } else {
            for (size_t idx = 0; idx < m_nr_label; ++idx) {
                for (size_t i = 0; i < m_nr_comps; ++i) {
                    m_components[idx * m_nr_comps + i] =
                        util::modulo<crt_val_t>(
                            (m_components[idx * m_nr_comps + i] +
                             l.m_components[i]),
                            m_modulus);
                }
            }
        }
    }

    void operator+=(const LabelSlice& s) {
        assert((m_nr_label == s.get_slice_size() || s.get_slice_size() == 1) &&
               "LabelTensor/Slice size missmatch");

        if (s.get_slice_size() == 1) {
            for (size_t i = 0; i < m_nr_label; ++i) {
                for (size_t j = 0; j < m_nr_comps; ++j) {
                    m_components[i * m_nr_comps + j] = util::modulo<crt_val_t>(
                        (m_components[i * m_nr_comps + j] +
                         s.get_components()[j]),
                        m_modulus);
                }
            }
        } else {
            size_t idx = 0;
            for (auto val = s.cbegin(); val != s.cend(); ++val) {
                for (size_t i = 0; i < m_nr_comps; ++i) {
                    m_components[idx * m_nr_comps + i] =
                        util::modulo<crt_val_t>(
                            (m_components[idx * m_nr_comps + i] + val[i]),
                            m_modulus);
                }
                ++idx;
            }
        }
    }

    void operator-=(const LabelTensor& l) {
        assert((m_nr_label == l.m_nr_label || l.m_nr_label == 1) &&
               "LabelTensor size missmatch");

        if (m_size == l.m_size) {
            for (size_t idx = 0; idx < m_nr_label; ++idx) {
                for (size_t i = 0; i < m_nr_comps; ++i) {
                    m_components[idx * m_nr_comps + i] =
                        util::modulo<crt_val_t>(
                            (m_components[idx * m_nr_comps + i] -
                             l.m_components[idx * m_nr_comps + i]),
                            m_modulus);
                }
            }
        } else {
            for (size_t idx = 0; idx < m_nr_label; ++idx) {
                for (size_t i = 0; i < m_nr_comps; ++i) {
                    m_components[idx * m_nr_comps + i] =
                        util::modulo<crt_val_t>(
                            (m_components[idx * m_nr_comps + i] -
                             l.m_components[i]),
                            m_modulus);
                }
            }
        }
    }

    void operator-=(const LabelSlice& s) {
        assert((m_nr_label == s.get_slice_size() || s.get_slice_size() == 1) &&
               "LabelTensor/Slice size missmatch");

        if (s.get_slice_size() == 1) {
            for (size_t i = 0; i < m_nr_label; ++i) {
                for (size_t j = 0; j < m_nr_comps; ++j) {
                    m_components[i * m_nr_comps + j] = util::modulo<crt_val_t>(
                        (m_components[i * m_nr_comps + j] -
                         s.get_components()[j]),
                        m_modulus);
                }
            }
        } else {
            size_t idx = 0;
            for (auto val = s.cbegin(); val != s.cend(); ++val) {
                for (size_t i = 0; i < m_nr_comps; ++i) {
                    m_components[idx * m_nr_comps + i] =
                        util::modulo<crt_val_t>(
                            (m_components[idx * m_nr_comps + i] - val[i]),
                            m_modulus);
                }
                ++idx;
            }
        }
    }

    template <typename T>
    void operator*=(const ScalarTensor<T>& s) {
        assert(m_nr_label == s.size() && "LabelTensor size missmatch");
        for (size_t idx = 0; idx < m_nr_label; ++idx) {
            for (size_t i = 0; i < m_nr_comps; ++i) {
                m_components[idx * m_nr_comps + i] = util::modulo<T, crt_val_t>(
                    (m_components[idx * m_nr_comps + i] * s.data()[idx]),
                    m_modulus);
            }
        }
    }

    void operator*=(const q_val_t c) {
        for (size_t i = 0; i < m_size; ++i) {
            m_components[i] = util::modulo<q_val_t, crt_val_t>(
                (m_components[i] * c), m_modulus);
        }
    }

    friend LabelTensor operator*(const LabelTensor& l, const q_val_t c) {
        LabelTensor l_result{l};
        l_result *= c;
        return l_result;
    }

    friend LabelTensor operator*(const q_val_t c, const LabelTensor& l) {
        return l * c;
    }

    // Label operations
    void compress() {
        if (m_compressed == nullptr) m_compressed = new __uint128_t[m_nr_label];
        for (size_t idx = 0; idx < m_nr_label; ++idx) {
            m_compressed[idx] = m_components[idx * m_nr_comps + m_nr_comps - 1];
            for (size_t i = m_nr_comps - 1; i > 0; --i) {
                m_compressed[idx] *= m_modulus;
                m_compressed[idx] += m_components[idx * m_nr_comps + i - 1];
            }
        }
    }

    void decompress() {
        for (size_t idx = 0; idx < m_nr_label; ++idx) {
            for (size_t i = 0; i < m_nr_comps; ++i) {
                // For this implicit cast to be valid, the modulus datatype must
                // be of the same bit-length as the components datatype
                m_components[idx * m_nr_comps + i] = m_compressed[idx];
                m_compressed[idx] /= m_modulus;
            }
            crt_val_t sub = 0;
            for (size_t i = m_nr_comps - 1; i > 0; --i) {
                sub -= m_components[idx * m_nr_comps + i] * m_modulus;
                m_components[idx * m_nr_comps + i - 1] += sub;
                sub *= m_modulus;
            }
            m_components[idx * m_nr_comps + m_nr_comps - 1] %= m_modulus;
        }
    }

    void fast_decompress() {
        for (size_t idx = 0; idx < m_nr_label; ++idx) {
            __uint128_t n = ipow(2, CHUNK_LENGTH);
            __uint128_t tmp;
            const uint16_t* lookup{get_decompress_lookup(m_modulus)};

            size_t offset = idx * m_nr_comps;
            memset(&m_components[offset], 0, m_nr_comps * sizeof(crt_val_t));

            for (size_t i = 0; i < NR_CHUNKS; i++) {
                tmp = m_compressed[idx] & ((n - 1) << (i * CHUNK_LENGTH));
                tmp = tmp >> (i * CHUNK_LENGTH);
                tmp += (tmp > 0) * (i << CHUNK_LENGTH);

                for (size_t j = m_nr_comps; j > 0; j--) {
                    m_components[offset + j - 1] +=
                        lookup[m_nr_comps * tmp + j - 1];
                }
            }

            // Mod-q addition
            uint32_t quot = 0;
            for (size_t j = 0; j < m_nr_comps; j++) {
                m_components[offset + j] += quot;

                quot = m_components[offset + j] / m_modulus;
                m_components[offset + j] = m_components[offset + j] % m_modulus;
            }
        }
    }

    void hash() {
        if (m_hashed == nullptr) m_hashed = new __uint128_t[m_nr_label];
        for (size_t idx = 0; idx < m_nr_label; ++idx) {
            m_hashed[idx] = m_cpu_aes_engine.cipher(&m_compressed[idx]);
        }
    }

    // Tensor operations
    static LabelTensor* matvecmul(const ScalarTensor<q_val_t>& m,
                                  LabelTensor& l, int nr_threads) {
        size_t input_dim = l.m_dims.at(0);
        size_t output_dim = m.get_dims().at(0);
        assert(m.get_dims().size() == 2 && "Dimension mismatch");
        assert(l.m_dims.size() == 1 && "Dimension mismatch");

        crt_val_t modulus = l.m_modulus;
        size_t nr_comps{get_nr_comps(modulus)};

        auto result = new LabelTensor{modulus, 0, dim_t{output_dim}};
        auto c_comps = result->get_components();
        auto b_comps = l.get_components();
        auto a_ptr = m.data();
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
        for (size_t j = 0; j < output_dim; ++j) {
            auto a = a_ptr + j * input_dim;
            auto c = c_comps + j * nr_comps;
            for (size_t k = 0; k < input_dim; ++k) {
                auto b = b_comps + k * nr_comps;
                for (size_t i = 0; i < nr_comps; ++i) {
                    c[i] = (c[i] + b[i] * a[k]) % modulus;
                }
            }
        }
        return result;
    }

    // Tensor operations
    static LabelTensor* matvecmul_zero(const ScalarTensor<q_val_t>& m,
                                       LabelTensor& l, LabelTensor zero,
                                       int nr_threads) {
        size_t input_dim = l.m_dims.at(0);
        size_t output_dim = m.get_dims().at(0);
        assert(m.get_dims().size() == 2 && "Dimension mismatch");
        assert(l.m_dims.size() == 1 && "Dimension mismatch");

        crt_val_t modulus = l.m_modulus;
        size_t nr_comps{get_nr_comps(modulus)};

        auto result = new LabelTensor{modulus, 0, dim_t{output_dim}};
        auto c_comps = result->get_components();
        auto b_comps = l.get_components();
        auto a_ptr = m.data();
        auto z_comps = zero.get_components();
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
        for (size_t j = 0; j < output_dim; ++j) {
            auto a = a_ptr + j * input_dim;
            auto c = c_comps + j * nr_comps;
            for (size_t k = 0; k < input_dim; ++k) {
                if (util::modulo(a[k], modulus) == 0) {
                    for (size_t i = 0; i < nr_comps; ++i)
                        c[i] = (c[i] + z_comps[i]) % modulus;
                } else {
                    auto b = b_comps + k * nr_comps;
                    for (size_t i = 0; i < nr_comps; ++i) {
                        c[i] = (c[i] + b[i] * a[k]) % modulus;
                    }
                }
            }
        }
        return result;
    }

    static LabelTensor* matvecmul_tf(const ScalarTensor<q_val_t>& m,
                                          LabelTensor& l, size_t channel,
                                          int nr_threads) {
        size_t input_dim = l.m_dims.at(0);
        size_t output_dim = m.get_dims().at(0);
        size_t offset = input_dim / channel;
        assert(m.get_dims().size() == 2 && "Dimension mismatch");
        assert(l.m_dims.size() == 1 && "Dimension mismatch");

        crt_val_t modulus = l.m_modulus;
        size_t nr_comps{get_nr_comps(modulus)};

        auto result = new LabelTensor{modulus, 0, dim_t{output_dim}};
        auto c_comps = result->get_components();
        auto b_comps = l.get_components();
        auto a_ptr = m.data();
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
        for (size_t j = 0; j < output_dim; ++j) {
            auto c = c_comps + j * nr_comps;
            auto a = a_ptr + j * input_dim;
            for (size_t k = 0; k < input_dim; ++k) {
                int input_idx = k / channel + (k % channel) * offset;
                auto b = b_comps + input_idx * nr_comps;
                for (size_t i = 0; i < nr_comps; ++i) {
                    c[i] = (c[i] + b[i] * a[k]) % modulus;
                }
            }
        }
        return result;
    }

    static LabelTensor* matvecmul_zero_tf(const ScalarTensor<q_val_t>& m,
                                     LabelTensor& l, LabelTensor zero,
                                     size_t channel, int nr_threads) {
        size_t input_dim = l.m_dims.at(0);
        size_t output_dim = m.get_dims().at(0);
        size_t offset = input_dim / channel;
        assert(m.get_dims().size() == 2 && "Dimension mismatch");
        assert(l.m_dims.size() == 1 && "Dimension mismatch");

        crt_val_t modulus = l.m_modulus;
        size_t nr_comps{get_nr_comps(modulus)};

        auto result = new LabelTensor{modulus, 0, dim_t{output_dim}};
        auto c_comps = result->get_components();
        auto b_comps = l.get_components();
        auto a_ptr = m.data();
        auto z_comps = zero.get_components();
#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
        for (size_t j = 0; j < output_dim; ++j) {
            auto c = c_comps + j * nr_comps;
            auto a = a_ptr + j * input_dim;
            for (size_t k = 0; k < input_dim; ++k) {
                if (util::modulo(a[k], modulus) == 0) {
                    for (size_t i = 0; i < nr_comps; ++i)
                        c[i] = (c[i] + z_comps[i]) % modulus;
                } else {
                    int input_idx = k / channel + (k % channel) * offset;
                    auto b = b_comps + input_idx * nr_comps;
                    for (size_t i = 0; i < nr_comps; ++i) {
                        c[i] = (c[i] + b[i] * a[k]) % modulus;
                    }
                }
            }
        }
        return result;
    }

    static LabelTensor* conv2d(LabelTensor& l, ScalarTensor<q_val_t>& weights,
                               LabelTensor& bias_label, size_t input_width,
                               size_t input_height, size_t channel,
                               size_t filter, size_t filter_width,
                               size_t filter_height, size_t stride_width,
                               size_t stride_height, int nr_threads) {
        size_t output_width = (input_width - filter_width) / stride_width + 1;
        size_t output_height =
            (input_height - filter_height) / stride_height + 1;

        size_t output_size = output_width * output_height;  // sizer per filter
        size_t filter_size = filter_width * filter_height;
        size_t input_size = input_width * input_height;

        size_t nr_comps = l.get_nr_comps();

        dim_t dims{output_height, output_width, filter};
        auto result = new LabelTensor{l.get_modulus(), 0, dims};

        crt_val_t* output = result->get_components();
        crt_val_t* input = l.get_components();
        crt_val_t modulus = l.get_modulus();

        q_val_t* we = weights.data();
        crt_val_t* bi = bias_label.get_components();
#ifndef SGX
#pragma omp parallel for collapse(4) num_threads(nr_threads)
#endif
        // applied on all label components
        for (size_t m = 0; m < nr_comps; ++m) {
            // for all filter
            for (size_t v = 0; v < filter; ++v) {
                // move filter along y-axis
                for (size_t l = 0; l < output_height; ++l) {
                    // move filter along x-axis
                    for (size_t k = 0; k < output_width; ++k) {
                        size_t output_idx = v * output_size * nr_comps +
                                            l * output_width * nr_comps +
                                            k * nr_comps + m;
                        // scalar product over all input-channel...
                        for (size_t w = 0; w < channel; ++w) {
                            // ...along filter height and...
                            for (size_t i = 0; i < filter_height; ++i) {
                                // ...width
                                for (size_t j = 0; j < filter_width; ++j) {
                                    output[output_idx] =
                                        (output[output_idx] +
                                         we[v * filter_size * channel +
                                            w * filter_size +
                                            i * filter_height + j] *
                                             input[w * input_size * nr_comps +
                                                   i * input_width * nr_comps +
                                                   j * nr_comps +
                                                   k * stride_width * nr_comps +
                                                   l * stride_height *
                                                       input_width * nr_comps +
                                                   m]) %
                                        modulus;
                                }
                            }
                        }
                        // add bias to each output-channel
                        output[output_idx] =
                            (output[output_idx] + bi[v * nr_comps + m]) %
                            modulus;
                    }
                }
            }
        }
        return result;
    }

    static LabelTensor* conv2d_zero(LabelTensor& l, LabelTensor zero,
                                    ScalarTensor<q_val_t>& weights,
                                    LabelTensor& bias_label, size_t input_width,
                                    size_t input_height, size_t channel,
                                    size_t filter, size_t filter_width,
                                    size_t filter_height, size_t stride_width,
                                    size_t stride_height, int nr_threads) {
        size_t output_width = (input_width - filter_width) / stride_width + 1;
        size_t output_height =
            (input_height - filter_height) / stride_height + 1;

        size_t output_size = output_width * output_height;  // sizer per filter
        size_t filter_size = filter_width * filter_height;
        size_t input_size = input_width * input_height;

        size_t nr_comps = l.get_nr_comps();

        dim_t dims{output_height, output_width, filter};
        auto result = new LabelTensor{l.get_modulus(), 0, dims};

        crt_val_t* output = result->get_components();
        crt_val_t* input = l.get_components();
        crt_val_t modulus = l.get_modulus();

        q_val_t* we = weights.data();
        crt_val_t* bi = bias_label.get_components();
        crt_val_t* z_comps = zero.get_components();
#ifndef SGX
#pragma omp parallel for collapse(4) num_threads(nr_threads)
#endif
        // applied on all label components
        for (size_t m = 0; m < nr_comps; ++m) {
            // for all filter
            for (size_t v = 0; v < filter; ++v) {
                // move filter along y-axis
                for (size_t l = 0; l < output_height; ++l) {
                    // move filter along x-axis
                    for (size_t k = 0; k < output_width; ++k) {
                        size_t output_idx = v * output_size * nr_comps +
                                            l * output_width * nr_comps +
                                            k * nr_comps + m;
                        // scalar product over all input-channel...
                        for (size_t w = 0; w < channel; ++w) {
                            // ...along filter height and...
                            for (size_t i = 0; i < filter_height; ++i) {
                                // ...width
                                for (size_t j = 0; j < filter_width; ++j) {
                                    q_val_t wei = we[v * filter_size * channel +
                                                     w * filter_size +
                                                     i * filter_height + j];
                                    if (util::modulo(wei, modulus) == 0) {
                                        output[output_idx] =
                                            (output[output_idx] + z_comps[m]) %
                                            modulus;
                                    } else {
                                        output[output_idx] =
                                            (output[output_idx] +
                                             wei * input[w * input_size *
                                                             nr_comps +
                                                         i * input_width *
                                                             nr_comps +
                                                         j * nr_comps +
                                                         k * stride_width *
                                                             nr_comps +
                                                         l * stride_height *
                                                             input_width *
                                                             nr_comps +
                                                         m]) %
                                            modulus;
                                    }
                                }
                            }
                        }
                        // add bias to each output-channel
                        output[output_idx] =
                            (output[output_idx] + bi[v * nr_comps + m]) %
                            modulus;
                    }
                }
            }
        }
        return result;
    }

    static LabelTensor* conv2d_static_bias_label(
        LabelTensor& l, ScalarTensor<q_val_t>& weights,
        const LabelSlice& bias_label, size_t input_width, size_t input_height,
        size_t channel, size_t filter, size_t filter_width,
        size_t filter_height, size_t stride_width, size_t stride_height) {
        size_t output_width = (input_width - filter_width) / stride_width + 1;
        size_t output_height =
            (input_height - filter_height) / stride_height + 1;

        size_t output_size = output_width * output_height;  // sizer per filter
        size_t filter_size = filter_width * filter_height;
        size_t input_size = input_width * input_height;

        size_t nr_comps = l.get_nr_comps();

        dim_t dims{output_height, output_width, filter};
        auto result = new LabelTensor{l.get_modulus(), 0, dims};

        crt_val_t* output = result->get_components();
        crt_val_t* input = l.get_components();
        crt_val_t modulus = l.get_modulus();

        q_val_t* we = weights.data();
        crt_val_t* bi = bias_label.get_components();
#ifndef SGX
#pragma omp parallel for collapse(4)
#endif
        // applied on all label components
        for (size_t m = 0; m < nr_comps; ++m) {
            // for all filter
            for (size_t v = 0; v < filter; ++v) {
                // move filter along y-axis
                for (size_t l = 0; l < output_height; ++l) {
                    // move filter along x-axis
                    for (size_t k = 0; k < output_width; ++k) {
                        size_t output_idx = v * output_size * nr_comps +
                                            l * output_width * nr_comps +
                                            k * nr_comps + m;
                        // scalar product over all input-channel...
                        for (size_t w = 0; w < channel; ++w) {
                            // ...along filter height and...
                            for (size_t i = 0; i < filter_height; ++i) {
                                // ...width
                                for (size_t j = 0; j < filter_width; ++j) {
                                    output[output_idx] =
                                        (output[output_idx] +
                                         we[v * filter_size * channel +
                                            w * filter_size +
                                            i * filter_height + j] *
                                             input[w * input_size * nr_comps +
                                                   i * input_width * nr_comps +
                                                   j * nr_comps +
                                                   k * stride_width * nr_comps +
                                                   l * stride_height *
                                                       input_width * nr_comps +
                                                   m]) %
                                        modulus;
                                }
                            }
                        }
                        // add bias to each output-channel
                        output[output_idx] =
                            (output[output_idx] + bi[m]) % modulus;
                    }
                }
            }
        }
        return result;
    }

    static LabelTensor* conv2d_static_bias_label_zero(
        LabelTensor& l, LabelTensor zero, ScalarTensor<q_val_t>& weights,
        const LabelSlice& bias_label, size_t input_width, size_t input_height,
        size_t channel, size_t filter, size_t filter_width,
        size_t filter_height, size_t stride_width, size_t stride_height) {
        size_t output_width = (input_width - filter_width) / stride_width + 1;
        size_t output_height =
            (input_height - filter_height) / stride_height + 1;

        size_t output_size = output_width * output_height;  // sizer per filter
        size_t filter_size = filter_width * filter_height;
        size_t input_size = input_width * input_height;

        size_t nr_comps = l.get_nr_comps();

        dim_t dims{output_height, output_width, filter};
        auto result = new LabelTensor{l.get_modulus(), 0, dims};

        crt_val_t* output = result->get_components();
        crt_val_t* input = l.get_components();
        crt_val_t modulus = l.get_modulus();

        q_val_t* we = weights.data();
        crt_val_t* bi = bias_label.get_components();
        crt_val_t* z_comps = zero.get_components();
#ifndef SGX
#pragma omp parallel for collapse(4)
#endif
        // applied on all label components
        for (size_t m = 0; m < nr_comps; ++m) {
            // for all filter
            for (size_t v = 0; v < filter; ++v) {
                // move filter along y-axis
                for (size_t l = 0; l < output_height; ++l) {
                    // move filter along x-axis
                    for (size_t k = 0; k < output_width; ++k) {
                        size_t output_idx = v * output_size * nr_comps +
                                            l * output_width * nr_comps +
                                            k * nr_comps + m;
                        // scalar product over all input-channel...
                        for (size_t w = 0; w < channel; ++w) {
                            // ...along filter height and...
                            for (size_t i = 0; i < filter_height; ++i) {
                                // ...width
                                for (size_t j = 0; j < filter_width; ++j) {
                                    q_val_t wei = we[v * filter_size * channel +
                                                     w * filter_size +
                                                     i * filter_height + j];
                                    if (util::modulo(wei, modulus) == 0) {
                                        output[output_idx] =
                                            (output[output_idx] + z_comps[m]) %
                                            modulus;
                                    } else {
                                        output[output_idx] =
                                            (output[output_idx] +
                                             wei * input[w * input_size *
                                                             nr_comps +
                                                         i * input_width *
                                                             nr_comps +
                                                         j * nr_comps +
                                                         k * stride_width *
                                                             nr_comps +
                                                         l * stride_height *
                                                             input_width *
                                                             nr_comps +
                                                         m]) %
                                            modulus;
                                    }
                                }
                            }
                        }
                        // add bias to each output-channel
                        output[output_idx] =
                            (output[output_idx] + bi[m]) % modulus;
                    }
                }
            }
        }
        return result;
    }

    crt_val_t get_modulus() const { return m_modulus; }
    void set_modulus(crt_val_t modulus) { m_modulus = modulus; }
    size_t get_nr_comps() const { return m_nr_comps; }
    dim_t get_dims() const { return m_dims; }
    void flatten() { m_dims = dim_t{m_nr_label}; }
    crt_val_t* get_components(size_t idx = 0) {
        return &m_components[idx * m_nr_comps];
    }
    size_t size() const { return m_size; }
    size_t get_nr_label() const { return m_nr_label; }
    __uint128_t* get_compressed(size_t idx = 0) {
        if (m_compressed != nullptr) return &m_compressed[idx];
        return nullptr;
    }
    __uint128_t* get_hashed(size_t idx = 0) {
        if (m_hashed != nullptr) return &m_hashed[idx];
        return nullptr;
    }
    LabelSlice get_label(size_t idx) { return LabelSlice{*this, idx}; }
    LabelSlice slice(slice_dim_t slice) { return LabelSlice{*this, slice}; }
    void set_label(LabelSlice v, size_t idx) {
        assert(v.get_modulus() == m_modulus && "Modulus mismatch");
        std::memcpy(&m_components[idx * m_nr_comps], v.get_components(),
                    sizeof(crt_val_t) * m_nr_comps);
    }
    void set_offset_value(crt_val_t value = 1) {
        for (size_t i = 0; i < m_nr_label; ++i) {
            m_components[i * m_nr_comps] = 1;
        }
    }

    void set_compressed(__uint128_t* compressed) { m_compressed = compressed; }

    void set_compressed(__uint128_t compressed, int idx) {
        if (m_compressed == nullptr) {
            m_compressed = new __uint128_t[m_nr_label];
        }
        m_compressed[idx] = compressed;
    }

    /**
     * @brief Get the color value of label
     *
     * @return T Color value
     */
    crt_val_t get_color(size_t i) const {
        assert(i < m_nr_label && "No color value, index out of range");
        return m_components[i * m_nr_comps];
    }

    /**
     * @brief Get label components at index
     *
     * @return vector<T> Components
     */
    vector<crt_val_t> get_components_vec(size_t idx = 0) const {
        return vector<crt_val_t>(m_components + idx * m_nr_comps,
                                 m_components + (idx + 1) * m_nr_comps);
    }

    /**
     * @brief Get label components
     *
     * @return vector<T> Components
     */
    vector<crt_val_t> as_vector() const {
        return vector<crt_val_t>(m_components, m_components + m_size);
    }

    /**
     * @brief Compute number of label components needed for a given modulus
     *
     * @param modulus
     * @return int Number of components
     */
    static size_t get_nr_comps(q_val_t modulus) {
        return static_cast<size_t>(floor(128 / log2(modulus)));
        // >128 bit can not be packed in a single __uint128_t
        // return static_cast<size_t>(ceil(128 / log2(modulus)));
    }

    void print_dim() const {
        printf("dims:");
        for (auto& d : m_dims) {
            printf("%lu ", d);
        }
        printf("\n");

        dim_t dim_cnt(m_dims.size(), 0);
        for (size_t i = 0; i < m_nr_label; ++i) {
            printf("[");
            for (size_t s = 0; s < m_nr_comps; ++s)
                printf("%d ", m_components[i * m_nr_comps + s]);
            printf("]");
            ++dim_cnt[0];
            size_t cnt = 0;  // pointer to print the right number of paragraphs
            for (size_t j = 0; j < m_dims.size() - 1; ++j) {
                if (dim_cnt[j] == m_dims[j]) {
                    if (cnt < 2) printf("\n");
                    cnt++;
                    dim_cnt[j] = 0;
                    ++dim_cnt[j + 1];
                }
            }
        }
        printf("\n");
    }

    void print() {
        for (size_t i = 0; i < m_nr_label; i++) {
            for (size_t j = 0; j < m_nr_comps; j++) {
                printf("%d ", m_components[i * m_nr_comps + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    void init_random() { init_rand(m_components, m_modulus, m_size); }

   private:
    static inline void init_rand(crt_val_t* vec, crt_val_t modulus,
                                 size_t nr_comps) {
        int max_value = pow(2, sizeof(__uint128_t)) - 1;
        for (size_t i = 0; i < nr_comps; ++i) {
            // do-loob needed to generate equally distributed random values
            // after modulo reduction
            do {
                while (!_rdrand16_step(reinterpret_cast<uint16_t*>(&vec[i]))) {
                }
            } while (vec[i] >= (max_value - max_value % modulus));
            vec[i] = util::modulo(vec[i], modulus);
        }
    }

    /**
     * @brief      Generate base-q representation of val.
     *
     * @param[in]  val     The value to represent as base-q value.
     * @param[in]  q       The modulus.
     * @param[in]  length  The length of the base-q representation.
     *
     * @return     Base-q representation of value.
     */
    static inline void red_mod(uint16_t* red, __uint128_t val, uint16_t q,
                               size_t label_length) {
        __uint128_t quot = val;

        for (size_t i = 0; i < label_length; i++) {
            red[i] = quot % q;
            quot /= q;
            if (quot == 0) break;
        }
    }

    /**
     * @brief      Interget exponentiation.
     *
     * @param[in]  base  The base.
     * @param[in]  exp   The exponent.
     *
     * @return     Power.
     */
    static inline __uint128_t ipow(__uint128_t base, __uint128_t exp) {
        __uint128_t result = 1;
        for (;;) {
            if (exp & 1) result *= base;
            exp >>= 1;
            if (!exp) break;
            base *= base;
        }

        return result;
    }

    static vector<vector<uint16_t>>& gen_decompress_lookup() {
        static vector<vector<uint16_t>> lookup;
        lookup.resize(MAX_LABEL_MODULO + 1);
        __uint128_t n = ipow(2, CHUNK_LENGTH);

        lookup.resize(MAX_LABEL_MODULO + 1);
        for (int i = 2; i <= MAX_LABEL_MODULO; ++i) {
            size_t label_length = get_nr_comps(i);
            lookup.at(i).resize(n * label_length * NR_CHUNKS);
            for (int k = 0; k < NR_CHUNKS; k++) {
#pragma omp parallel for
                for (__uint128_t j = 0; j < n; j++) {
                    red_mod(&lookup.at(i).at(k * n * label_length +
                                             j * label_length),
                            j << (k * CHUNK_LENGTH), i, label_length);
                }
            }
        }
        return lookup;
    }
};

///////////////////////////////
// LabelSlice Implementation //
///////////////////////////////
LabelSlice::LabelSlice(LabelTensor& tensor, size_t label_idx)
    : m_tensor(tensor), m_label_ids{label_idx}, m_slice_size{1} {
    assert(label_idx < tensor.get_nr_label() && "Label index out of range");
    // restore slice dims from label index
    size_t nr_dims = tensor.get_dims().size();
    m_slice_dims.resize(nr_dims);
    size_t tmp = label_idx;
    for (size_t i = 0; i < nr_dims; ++i) {
        m_slice_dims[i] = std::make_pair(tmp % tensor.get_dims()[i],
                                         tmp % tensor.get_dims()[i]);
        tmp /= tensor.get_dims()[i];
    }
}

LabelSlice::LabelSlice(LabelTensor& tensor, slice_dim_t& slice_idx)
    : m_tensor(tensor), m_slice_dims(slice_idx) {
    assert(slice_idx.size() == m_tensor.get_dims().size() &&
           "Slice dimensions does not match tensor dimension");
    m_slice_size =
        std::accumulate(slice_idx.begin(), slice_idx.end(), (size_t)1,
                        [](size_t a, std::pair<size_t, size_t> b) {
                            return a * (b.second + 1 - b.first);
                        });
}

// Getter
LabelTensor& LabelSlice::get_tensor() const { return m_tensor; }

crt_val_t LabelSlice::get_modulus() const { return m_tensor.get_modulus(); }

size_t LabelSlice::get_nr_comps() const { return m_tensor.get_nr_comps(); }

crt_val_t* LabelSlice::get_components() const {
    assert(m_slice_size == 1 &&
           "Slice size greater than 0, label does not lie consecutive in "
           "memory");
    return m_tensor.get_components(m_label_ids.at(0));
}

__uint128_t* LabelSlice::get_compressed() const {
    assert(m_slice_size == 1 &&
           "Slice size greater than 0, label does not lie consecutive in "
           "memory");
    return m_tensor.get_compressed(m_label_ids.at(0));
}

__uint128_t* LabelSlice::get_hashed() const {
    assert(m_slice_size == 1 &&
           "Slice size greater than 0, label does not lie consecutive in "
           "memory");
    return m_tensor.get_hashed(m_label_ids.at(0));
}

crt_val_t LabelSlice::get_color() const {
    assert(m_slice_size == 1 &&
           "Slice size greater than 0, label does not lie consecutive in "
           "memory");
    return m_tensor.get_color(m_label_ids.at(0));
}

size_t LabelSlice::get_slice_size() const { return m_slice_size; }

slice_dim_t LabelSlice::get_slice_dims() const { return m_slice_dims; }

// Basic Operations
void LabelSlice::operator+=(const LabelSlice& l) {
    assert(m_slice_size == l.m_slice_size &&
           "LabelSlice size missmatch, cannot add");
    auto a = this->begin();
    auto b = l.cbegin();
    for (size_t i = 0; i < m_slice_size; ++i) {
        for (size_t j = 0; j < get_nr_comps(); ++j) {
            a[j] = util::modulo<q_val_t>((a[j] + b[j]), get_modulus());
        }
        ++a;
        ++b;
    }
}

void LabelSlice::operator-=(const LabelSlice& l) {
    assert(m_slice_size == l.m_slice_size &&
           "LabelSlice size missmatch, cannot subtract");
    auto a = this->begin();
    auto b = l.cbegin();
    for (size_t i = 0; i < m_slice_size; ++i) {
        for (size_t j = 0; j < get_nr_comps(); ++j) {
            a[j] = util::modulo<q_val_t>((a[j] - b[j]), get_modulus());
        }
        ++a;
        ++b;
    }
}

void LabelSlice::operator*=(const q_val_t c) {
    for (auto comp : *this) {
        for (size_t i = 0; i < get_nr_comps(); ++i) {
            comp[i] = util::modulo<q_val_t>((comp[i] * c), get_modulus());
        }
    }
}

// Friends
LabelTensor operator*(const LabelSlice& l, const q_val_t c) {
    LabelTensor l_result{l};
    l_result *= c;
    return l_result;
}

LabelTensor operator*(const q_val_t c, const LabelSlice& l) { return l * c; }

LabelTensor operator+(const LabelSlice& l1, const LabelSlice& l2) {
    LabelTensor l_result{l1};
    l_result += l2;
    return l_result;
}

LabelTensor operator-(const LabelSlice& l1, const LabelSlice& l2) {
    LabelTensor l_result{l1};
    l_result -= l2;
    return l_result;
}

// MISC
void LabelSlice::print() const {
    for (auto val = this->cbegin(); val != this->cend(); ++val) {
        for (size_t i = 0; i < get_nr_comps(); ++i) {
            printf("%d ", val[i]);
        }
        printf("\n");
    }
}

// Iterator
LabelSlice::Iterator LabelSlice::begin() { return Iterator(*this); }

LabelSlice::Iterator LabelSlice::end() {
    vector<size_t> slice_idx(m_slice_dims.size(), 0);
    slice_idx.back() = m_slice_dims.back().second + 1;
    return Iterator(*this, slice_idx);
}

LabelSlice::ConstIterator LabelSlice::cbegin() const {
    return ConstIterator(*this);
}

LabelSlice::ConstIterator LabelSlice::cend() const {
    vector<size_t> slice_idx(m_slice_dims.size(), 0);
    slice_idx.back() = m_slice_dims.back().second + 1;
    return ConstIterator(*this, slice_idx);
}

#endif
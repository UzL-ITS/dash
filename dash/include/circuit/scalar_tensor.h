#ifndef SCALAR_TENSOR_H
#define SCALAR_TENSOR_H

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#ifdef SGX
#include "misc/enclave_functions.h"
#endif

#include "misc/datatypes.h"

using std::vector;

template <typename T>
class ScalarTensor {
    dim_t m_dims;
    size_t m_size;
    size_t m_capacity;
    T* m_values;

   public:
    ScalarTensor() {
        m_values = nullptr;
        m_size = 0;
        m_dims = dim_t{0};
    }

    ScalarTensor(dim_t dims) {
        m_size = 0;
        m_dims = dims;
        m_capacity = std::accumulate(m_dims.begin(), m_dims.end(), 1,
                                     std::multiplies<size_t>());
        m_values = (T*)malloc(m_capacity * sizeof(T));
    }

    ScalarTensor(std::initializer_list<T> values, dim_t dims) {
        m_dims = dims;
        m_capacity = std::accumulate(m_dims.begin(), m_dims.end(), 1,
                                     std::multiplies<size_t>());
        assert(values.size() == m_capacity &&
               "ScalarTensor: initializer_list size mismatch");
        m_size = m_capacity;
        m_values = (T*)malloc(m_capacity * sizeof(T));
        std::memcpy(m_values, values.begin(), m_capacity * sizeof(T));
    }

    ScalarTensor(T* values, dim_t dims) {
        m_dims = dims;
        m_size = std::accumulate(m_dims.begin(), m_dims.end(), 1,
                                 std::multiplies<size_t>());
        m_capacity = m_size;
        m_values = (T*)malloc(m_size * sizeof(T));
        std::memcpy(m_values, values, m_size * sizeof(T));
    }

    ScalarTensor(T value, dim_t dims) {
        m_dims = dims;
        m_size = std::accumulate(m_dims.begin(), m_dims.end(), 1,
                                 std::multiplies<size_t>());
        m_capacity = m_size;
        m_values = (T*)malloc(m_capacity * sizeof(T));
        for (size_t i = 0; i < m_size; ++i) {
            m_values[i] = value;
        }
    }

    ScalarTensor(vector<T> values, dim_t dims) {
        m_dims = dims;
        m_size = std::accumulate(m_dims.begin(), m_dims.end(), 1,
                                 std::multiplies<size_t>());
        assert(values.size() == m_size && "ScalarTensor: vector size mismatch");
        m_capacity = m_size;
        m_values = (T*)malloc(m_capacity * sizeof(T));
        std::memcpy(m_values, values.data(), m_size * sizeof(T));
    }

    ScalarTensor(const ScalarTensor<T>& other) {
        m_values = (T*)malloc(other.m_size * sizeof(T));
        std::memcpy(m_values, other.data(), other.size() * sizeof(T));
        m_dims = other.m_dims;
        m_size = other.m_size;
        m_capacity = other.m_capacity;
    }

    ScalarTensor(ScalarTensor<T>&& other) noexcept {
        m_values = other.m_values;
        m_dims = other.m_dims;
        m_size = other.m_size;
        m_capacity = other.m_capacity;
        other.m_values = nullptr;
    }

    ~ScalarTensor() { free(m_values); }

    template <typename V>
    static ScalarTensor<T> create_with_cast(V* values, dim_t dims) {
        ScalarTensor<T> tensor(dims);
        for (size_t i = 0; i < tensor.get_capacity(); ++i) {
            tensor.push_back(static_cast<V>(values[i]));
        }
        return tensor;
    }

    // Contains code from
    // https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
    void transpose() {
        assert(m_size > 0 && "ScalarTensor: transpose on empty tensor");
        assert(m_dims.size() == 2 && "ScalarTensor: transpose on non-matrix");
        auto t_values = (T*)malloc(m_size * sizeof(T));

        for (size_t n = 0; n < m_dims[0] * m_dims[1]; n++) {
            size_t i = n / m_dims[0];
            size_t j = n % m_dims[0];
            t_values[n] = m_values[m_dims[1] * j + i];
        }
        free(m_values);
        m_values = t_values;
        std::swap(m_dims[0], m_dims[1]);
    }

    ScalarTensor<T>& operator=(const ScalarTensor<T>& that) {
        if (this != &that) {
            free(m_values);
            m_values = (T*)malloc(that.m_size * sizeof(T));
            std::memcpy(m_values, that.data(), that.m_size * sizeof(T));
            m_dims = that.m_dims;
            m_size = that.m_size;
            m_capacity = that.m_capacity;
        }
        return *this;
    }

    ScalarTensor<T>& operator=(ScalarTensor<T>&& that) {
        if (this != &that) {
            free(m_values);
            m_values = that.m_values;
            m_dims = that.m_dims;
            m_size = that.m_size;
            m_capacity = that.m_capacity;
            that.m_values = nullptr;
        }
        return *this;
    }

    T max() const {
        T max = m_values[0];
        for (size_t i = 1; i < m_size; ++i) {
            if (m_values[i] > max) {
                max = m_values[i];
            }
        }
        return max;
    }

    size_t argmax() const {
        T max = m_values[0];
        size_t argmax = 0;
        for (size_t i = 1; i < m_size; ++i) {
            if (m_values[i] > max) {
                max = m_values[i];
                argmax = i;
            }
        }
        return argmax;
    }

    T min() const {
        T min = m_values[0];
        for (size_t i = 1; i < m_size; ++i) {
            if (m_values[i] < min) {
                min = m_values[i];
            }
        }
        return min;
    }

    size_t argmin() const {
        T min = m_values[0];
        size_t argmin = 0;
        for (size_t i = 1; i < m_size; ++i) {
            if (m_values[i] < min) {
                min = m_values[i];
                argmin = i;
            }
        }
        return argmin;
    }

    T at(size_t i) const {
        assert(i < m_size && "Index out of bounds");
        return m_values[i];
    }

    T at(std::initializer_list<size_t> indices) const {
        assert(indices.size() == m_dims.size() &&
               "ScalarTensor: number of indices does not match number of "
               "dimensions");
        size_t index = 0;
        size_t stride = 1;
        auto it = indices.begin();
        for (size_t i = 0; i < m_dims.size(); ++i) {
            assert(*it < m_dims[i] && "Index out of bounds");
            index += *it * stride;
            stride *= m_dims[i];
            ++it;
        }
        return m_values[index];
    }

    T operator[](size_t index) const { return m_values[index]; }

    bool operator==(const ScalarTensor<T>& other) const {
        if (m_size != other.m_size) {
            return false;
        }
        for (size_t i = 0; i < m_size; ++i) {
            if (m_values[i] != other.m_values[i]) {
                return false;
            }
        }
        return true;
    }

    T* data() const { return m_values; }

    T back() const { return m_values[m_size - 1]; }

    void reserve(size_t capacity) {
        m_capacity = capacity;
        m_values =
            static_cast<T*>(std::realloc(m_values, m_capacity * sizeof(T)));
    }

    void resize(dim_t dims) {
        m_dims = dims;
        m_size = 1;
        for (auto& dim : m_dims) {
            m_size *= dim;
        }
        m_capacity = m_size;
        m_values =
            static_cast<T*>(std::realloc(m_values, m_capacity * sizeof(T)));
    }

    void push_back(T value) {
        assert(m_size < m_capacity);
        m_values[m_size] = value;
        ++m_size;
    }

    void set(dim_t dim, T value) {
        size_t idx = 0;
        size_t idx_prod = 1;
        for (size_t i = 0; i < dim.size(); ++i) {
            idx += dim[i] * idx_prod;
            idx_prod *= m_dims[i];
        }
        assert(idx < m_capacity && "Index out of bounds");
        m_values[idx] = value;
        m_size = std::max(m_size, idx + 1);
    }

    void set(std::initializer_list<size_t> indices, T value) {
        assert(indices.size() == m_dims.size() &&
               "ScalarTensor: number of indices does not match number of "
               "dimensions");
        size_t index = 0;
        size_t stride = 1;
        auto it = indices.begin();
        for (size_t i = 0; i < m_dims.size(); ++i) {
            assert(*it < m_dims[i] && "Index out of bounds");
            index += *it * stride;
            stride *= m_dims[i];
            ++it;
        }
        assert(index < m_capacity && "Index out of bounds");
        m_values[index] = value;
        m_size = std::max(m_size, index + 1);
    }

    T get(dim_t dim) const {
        int idx = 0;
        int idx_prod = 1;
        for (size_t i = 0; i < dim.size(); ++i) {
            idx += dim[i] * idx_prod;
            idx_prod *= m_dims[i];
        }
        assert(idx < m_size && "Index out of bounds");
        return m_values[idx];
    }

    dim_t get_dims() const { return m_dims; }
    void set_dims(dim_t dims) { m_dims = dims; }
    size_t size() const { return m_size; }
    size_t get_capacity() const { return m_capacity; }

    /**
     * @brief Scale given values with given modulus.
     *
     * @param dividend
     */
    void mod(T dividend) {
        for (size_t i = 0; i < m_size; ++i)
            m_values[i] = (m_values[i] % dividend + dividend) % dividend;
    }

    vector<T> as_vector() const {
        vector<T> v(m_values, m_values + m_size);
        return v;
    }

    void map(T (*functionality)(T)) {
        for (size_t i = 0; i < m_size; ++i) {
            m_values[i] = functionality(m_values[i]);
        }
    }

    // Slow AF, but works :)
    ScalarTensor<T> matvecmul(ScalarTensor<T> vec, T* min, T* max) const {
        size_t input_dim = vec.get_dims().at(0);
        size_t output_dim = m_dims.at(0);
        ScalarTensor<T> result((T)0, dim_t{output_dim});
        for (size_t i = 0; i < output_dim; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                result.data()[i] += m_values[i * input_dim + j] * vec.at(j);
                *min = std::min(*min, result.data()[i]);
                *max = std::max(*max, result.data()[i]);
            }
        }
        return result;
    }

    ScalarTensor<T> matvecmul(ScalarTensor<T> vec) const {
        size_t input_dim = vec.get_dims().at(0);
        size_t output_dim = m_dims.at(0);
        ScalarTensor<T> result((T)0, dim_t{output_dim});
        for (size_t i = 0; i < output_dim; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                result.data()[i] += m_values[i * input_dim + j] * vec.at(j);
            }
        }
        return result;
    }

    ScalarTensor<T> matvecmul_tf(ScalarTensor<T> vec, T* min, T* max,
                                 int channel) const {
        size_t input_dim = vec.get_dims().at(0);
        size_t output_dim = m_dims.at(0);
        size_t offset = input_dim / channel;
        ScalarTensor<T> result((T)0, dim_t{output_dim});
        for (size_t i = 0; i < output_dim; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                size_t input_idx = j / channel + (j % channel) * offset;
                result.data()[i] +=
                    m_values[i * input_dim + j] * vec.at(input_idx);
                *min = std::min(*min, result.data()[i]);
                *max = std::max(*max, result.data()[i]);
            }
        }
        return result;
    }

    ScalarTensor<T> matvecmul_tf(ScalarTensor<T> vec, int channel) const {
        size_t input_dim = vec.get_dims().at(0);
        size_t output_dim = m_dims.at(0);
        size_t offset = input_dim / channel;
        ScalarTensor<T> result((T)0, dim_t{output_dim});
        for (size_t i = 0; i < output_dim; ++i) {
            for (size_t j = 0; j < input_dim; ++j) {
                size_t input_idx = j / channel + (j % channel) * offset;
                result.data()[i] +=
                    m_values[i * input_dim + j] * vec.at(input_idx);
            }
        }
        return result;
    }

    ScalarTensor<T> max_pool(size_t kernel_width, size_t kernel_height,
                             size_t stride_width = 1,
                             size_t stride_height = 1) const {
        size_t input_width = m_dims.at(0);
        size_t input_height = m_dims.at(1);
        size_t input_channel = m_dims.at(2);
        size_t output_width = (input_width - kernel_width) / stride_width + 1;
        size_t output_height =
            (input_height - kernel_height) / stride_height + 1;
        size_t output_channel = input_channel;

        ScalarTensor<T> result(
            dim_t{output_width, output_height, output_channel});

        for (size_t i = 0; i < output_width; ++i) {
            for (size_t j = 0; j < output_height; ++j) {
                for (size_t k = 0; k < output_channel; ++k) {
                    T max = this->at({i * stride_width, j * stride_height, k});
                    for (size_t l = 0; l < kernel_width; ++l) {
                        for (size_t m = 0; m < kernel_height; ++m) {
                            max = std::max(
                                max, this->at({i * stride_width + l,
                                               j * stride_height + m, k}));
                        }
                    }
                    result.set({i, j, k}, max);
                }
            }
        }

        return result;
    }

    void operator+=(const ScalarTensor<T>& other) {
        assert(m_size == other.m_size && "Tensor size mismatch");
        for (size_t i = 0; i < m_size; ++i) {
            m_values[i] += other.m_values[i];
        }
    }

    void operator*=(T scalar) {
        for (size_t i = 0; i < m_size; ++i) {
            m_values[i] *= scalar;
        }
    }

    void operator/=(T scalar) {
        for (size_t i = 0; i < m_size; ++i) {
            m_values[i] /= scalar;
        }
    }

    ScalarTensor<T> operator/(T scalar) {
        ScalarTensor<T> result(*this);
        result /= scalar;
        return result;
    }

    /**
     * @brief Quantize the given values.
     *
     * Divide given values by quantization constant and round to nearest value
     * of integer type T.
     *
     * @param values
     * @param q_const
     * @return ScalarTensor<T>
     */
    static ScalarTensor<T> quantize(ScalarTensor<wandb_t>& values,
                                    QuantizationMethod q_method,
                                    wandb_t q_const) {
        assert(q_method == QuantizationMethod::SimpleQuant &&
               "SimpleQuant requires a quantization constant of type wandb_t.");
        ScalarTensor<T> quantized_values{values.get_dims()};
        for (size_t i = 0; i < values.size(); ++i) {
            quantized_values.push_back(std::llround(values.at(i) / q_const));
        }
        return quantized_values;
    }

    static ScalarTensor<T> quantize(ScalarTensor<wandb_t>& values,
                                    QuantizationMethod q_method, int l) {
        assert(q_method == QuantizationMethod::ScaleQuant &&
               "ScaleQuant requires a quantization constant of type int.");
        ScalarTensor<T> result{values.get_dims()};
        for (size_t i = 0; i < values.size(); ++i) {
            result.push_back(std::llround(values.at(i) * std::pow(2, l)));
        }
        return result;
    }

    static ScalarTensor<T> rescale(ScalarTensor<q_val_t>& values, int l) {
        ScalarTensor<T> result{values.get_dims()};
        for (size_t i = 0; i < values.size(); ++i) {
            // imitate rounding behaviour of garbled rescaling
            result.push_back(std::ceil(values.at(i) * std::pow(2, -l)));
        }
        return result;
    }

    void print() const {
        std::string ss = "";
        if (typeid(T) == typeid(int)) {
            ss.append("%d ");
        } else if (typeid(T) == typeid(float)) {
            ss.append("%f ");
        } else if (typeid(T) == typeid(double)) {
            ss.append("%lf ");
        } else if (typeid(T) == typeid(long)) {
            ss.append("%ld ");
        } else if (typeid(T) == typeid(long long)) {
            ss.append("%lld ");
        } else if (typeid(T) == typeid(unsigned int)) {
            ss.append("%u ");
        } else if (typeid(T) == typeid(unsigned long)) {
            ss.append("%lu ");
        } else if (typeid(T) == typeid(unsigned long long)) {
            ss.append("%llu ");
        } else {
            ss.append("%d ");
        }

        auto s = ss.c_str();
        if (m_dims.size() <= 2) {
            size_t dim2 = (m_dims.size() == 2) ? m_dims.at(1) : 1;
            for (size_t i = 0; i < m_dims.at(0); ++i) {
                for (size_t j = 0; j < dim2; ++j) {
                    printf(s, m_values[i * dim2 + j]);
                }
                printf("\n");
            }
        } else {
            dim_t dim_cnt(m_dims.size(), 0);
            for (size_t i = 0; i < m_size; ++i) {
                printf(s, m_values[i]);
                ++dim_cnt[0];
                int cnt = 0;  // pointer to print the right number of paragraphs
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
    }
};

#endif
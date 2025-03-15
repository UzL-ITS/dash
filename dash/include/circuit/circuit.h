#ifndef CIRCUIT_H
#define CIRCUIT_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <vector>

#include "circuit/layer/layer.h"
#include "circuit/scalar_tensor.h"
#include "misc/datatypes.h"
#include "misc/util.h"

using std::vector;

class Circuit {
    dim_t m_input_dims;
    dim_t m_output_dims;
    size_t m_input_size;
    size_t m_output_size;
    vector<Layer*> m_layer;

    int m_ql;
    std::vector<int> m_qs;

   public:
    Circuit(std::initializer_list<Layer*> layer, int ql = 5, std::vector<int> qs = {2, 17})
        : m_ql{ql}, m_qs{qs} {
        m_layer.reserve(layer.size());
        for (size_t i = 0; i < layer.size(); i++) {
            m_layer.push_back(layer.begin()[i]);
        }
        m_input_dims = m_layer[0]->get_input_dims();
        m_output_dims = m_layer[m_layer.size() - 1]->get_output_dims();
        m_input_size =
            std::accumulate(m_input_dims.begin(), m_input_dims.end(), 1lu,
                            [](size_t a, size_t b) { return a * b; });
        m_output_size =
            std::accumulate(m_output_dims.begin(), m_output_dims.end(), 1lu,
                            [](size_t a, size_t b) { return a * b; });
    }

    Circuit(vector<Layer*> layer, int ql = 5, std::vector<int> qs = {2, 17})
        : m_layer{layer}, m_ql{ql}, m_qs{qs} {
        m_input_dims = m_layer.at(0)->get_input_dims();
        m_output_dims = m_layer[m_layer.size() - 1]->get_output_dims();
        m_input_size =
            std::accumulate(m_input_dims.begin(), m_input_dims.end(), 1lu,
                            [](size_t a, size_t b) { return a * b; });
        m_output_size =
            std::accumulate(m_output_dims.begin(), m_output_dims.end(), 1lu,
                            [](size_t a, size_t b) { return a * b; });
    }

    ~Circuit() {
        for (size_t i = 0; i < m_layer.size(); i++) {
            delete m_layer.at(i);
        }
    }

    ScalarTensor<wandb_t> plain_eval(ScalarTensor<wandb_t> input,
                                     bool track_extreme_values = true) {
        assert(input.size() == m_input_size &&
               "Input size does not match circuit input size");

        ScalarTensor<wandb_t> output = input;
        for (size_t i = 0; i < m_layer.size(); i++) {
            output = m_layer.at(i)->plain_eval(output, track_extreme_values);
        }
        return output;
    }

    vector<ScalarTensor<wandb_t>> plain_eval(
        vector<ScalarTensor<wandb_t>> inputs,
        bool track_extreme_values = true) {
        vector<ScalarTensor<wandb_t>> outputs;
        outputs.resize(inputs.size());
        // #pragma omp parallel for
        for (size_t i = 0; i < inputs.size(); i++)
            outputs.at(i) = plain_eval(inputs.at(i), track_extreme_values);
        return outputs;
    }

    double plain_test(vector<ScalarTensor<wandb_t>> inputs,
                      vector<unsigned long> labels) {
        double test_accuracy = 0;
#pragma omp parallel for reduction(+ : test_accuracy)
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto outputs = plain_eval(inputs.at(i), false);
            auto infered_label = outputs.argmax();
            if (infered_label == labels.at(i)) {
                test_accuracy += 1;
            }
        }
        return test_accuracy / inputs.size();
    }

    ScalarTensor<q_val_t> plain_q_eval(ScalarTensor<q_val_t> input,
                                       bool track_extreme_values = true) {
        assert(input.size() == m_input_size &&
               "Input size does not match circuit input size");
        ScalarTensor<q_val_t> output = input;
        for (size_t i = 0; i < m_layer.size(); i++) {
            output = m_layer.at(i)->plain_q_eval(output, track_extreme_values);
        }
        return output;
    }

    vector<ScalarTensor<q_val_t>> plain_q_eval(
        vector<ScalarTensor<q_val_t>> inputs,
        bool track_extreme_values = true) {
        vector<ScalarTensor<q_val_t>> outputs;
        outputs.resize(inputs.size());

        for (size_t i = 0; i < inputs.size(); i++) {
            outputs.at(i) = plain_q_eval(inputs.at(i), track_extreme_values);
        }
        return outputs;
    }

    double plain_q_test(vector<ScalarTensor<q_val_t>> inputs,
                        vector<unsigned long> labels) {
        double test_accuracy = 0;
#pragma omp parallel for reduction(+ : test_accuracy)
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto outputs = plain_q_eval(inputs.at(i), false);
            auto infered_label = outputs.argmax();
            if (infered_label == labels.at(i)) {
                test_accuracy += 1;
            }
        }
        return test_accuracy / inputs.size();
    }

    double compute_q_acc(ScalarTensor<wandb_t> input,
                         ScalarTensor<q_val_t> q_input, wandb_t q_constant,
                         double error_bound = 1.0) {
        ScalarTensor<wandb_t> output = plain_eval(input);
        ScalarTensor<q_val_t> q_output = plain_q_eval(q_input);

        // reconstruct real output from quantized output
        auto rec = ScalarTensor<wandb_t>::create_with_cast<q_val_t>(
            q_output.data(), q_output.get_dims());
        rec *= q_constant;

        int correct = 0;
        for (size_t i = 0; i < output.size(); i++) {
            wandb_t diff = std::abs(output.at(i) - rec.at(i));
            wandb_t relative_error = diff / std::abs(output.at(i));
            if (relative_error < error_bound) {
                correct++;
            }
        }
        return (double)correct / output.size();
    }

    int infer_crt_base_size(ScalarTensor<q_val_t> input) {
        plain_q_eval(input);
        // minimal needed crt modulus needed to garble circuit with given input
        //// over all layer
        q_val_t min_plain_q_val{Q_VAL_MAX};
        q_val_t max_plain_q_val{Q_VAL_MIN};

        for (auto layer : m_layer) {
            min_plain_q_val =
                std::min(min_plain_q_val, layer->get_min_plain_q_val());
            max_plain_q_val =
                std::max(max_plain_q_val, layer->get_max_plain_q_val());
        }

        q_val_t min_crt_modulus =
            2 * std::max(abs(min_plain_q_val), abs(max_plain_q_val));

        // find smallest crt modulus that is larger than min_crt_modulus
        q_val_t crt_modulus = 0;
        vector<crt_val_t> crt_base;
        int crt_base_size = 0;

        while (crt_modulus < min_crt_modulus) {
            crt_base_size++;
            assert(crt_base_size <= 11 &&
                   "Infered crt_base_size too large, optimize quantization "
                   "constant.");
            crt_base = util::sieve_of_eratosthenes<crt_val_t>(crt_base_size);
            crt_modulus = std::accumulate(begin(crt_base), end(crt_base), 1,
                                          [](int a, int b) { return a * b; });
        }
        assert(crt_base_size > 0 &&
               "Infered crt_base_size too small, optimize quantization "
               "constant.");
        return crt_base_size;
    }

    int infer_crt_base_size(vector<ScalarTensor<q_val_t>> inputs) {
        plain_q_eval(inputs);
        // minimal needed crt modulus needed to garble circuit with given input
        //// over all layer
        q_val_t min_plain_q_val{Q_VAL_MAX};
        q_val_t max_plain_q_val{Q_VAL_MIN};

        for (auto layer : m_layer) {
            min_plain_q_val =
                std::min(min_plain_q_val, layer->get_min_plain_q_val());
            max_plain_q_val =
                std::max(max_plain_q_val, layer->get_max_plain_q_val());
        }

        q_val_t min_crt_modulus =
            2 * std::max(abs(min_plain_q_val), abs(max_plain_q_val));

        // find smallest crt modulus that is larger than min_crt_modulus
        q_val_t crt_modulus = 0;
        vector<crt_val_t> crt_base;
        int crt_base_size = 0;

        while (crt_modulus < min_crt_modulus) {
            crt_base_size++;
            assert(crt_base_size <= 11 &&
                   "Infered crt_base_size too large, optimize quantization "
                   "constant.");
            crt_base = util::sieve_of_eratosthenes<crt_val_t>(crt_base_size);
            crt_modulus = std::accumulate(begin(crt_base), end(crt_base), 1,
                                          [](int a, int b) { return a * b; });
        }
        assert(crt_base_size > 0 &&
               "Infered crt_base_size too small, optimize quantization "
               "constant.");
        return crt_base_size;
    }

    int infer_crt_base_size_no_assert(vector<ScalarTensor<q_val_t>> inputs) {
        plain_q_eval(inputs, true);
        // minimal needed crt modulus needed to garble circuit with given input
        //// over all layer
        q_val_t min_plain_q_val{Q_VAL_MAX};
        q_val_t max_plain_q_val{Q_VAL_MIN};

        for (auto layer : m_layer) {
            min_plain_q_val =
                std::min(min_plain_q_val, layer->get_min_plain_q_val());
            max_plain_q_val =
                std::max(max_plain_q_val, layer->get_max_plain_q_val());
        }

        q_val_t min_crt_modulus =
            2 * std::max(abs(min_plain_q_val), abs(max_plain_q_val));

        // find smallest crt modulus that is larger than min_crt_modulus
        q_val_t crt_modulus = 0;
        vector<crt_val_t> crt_base;
        int crt_base_size = 0;

        while (crt_modulus < min_crt_modulus) {
            if (crt_base_size > 100) {
                return -1;
            }
            crt_base_size++;
            crt_base = util::sieve_of_eratosthenes<crt_val_t>(crt_base_size);
            crt_modulus = std::accumulate(begin(crt_base), end(crt_base), 1,
                                          [](int a, int b) { return a * b; });
        }
        return crt_base_size;
    }

    void quantize(wandb_t q_const) {
        for (auto layer : m_layer) {
            layer->quantize(q_const);
        }
    }

    void optimize_quantization(int target_crt_base_size,
                               vector<ScalarTensor<wandb_t>> inputs,
                               double init_q_val = 0.2,
                               double init_step_width = 0.01,
                               double final_step_width = 0.00001,
                               int nr_samples = -1) {
        double step_width = init_step_width;
        int crt_base_size = -1;
        double q_val = init_q_val;
        double last_q_val = init_q_val;
        quantize(q_val);

        printf("Search quantization constant\n");
        while (crt_base_size != target_crt_base_size ||
               step_width > final_step_width) {
            assert(q_val != 0 && "q_val cannot be 0");

            auto quantized_inputs = vec_quantize(inputs, q_val, nr_samples);
            crt_base_size = infer_crt_base_size_no_assert(quantized_inputs);
            if (crt_base_size == -1) {
                quantize(last_q_val);
                return;
            }
            printf("q_val: %.15f, crt_base_size: %d\n", q_val, crt_base_size);
            if (crt_base_size > target_crt_base_size) {
                if (is_equal(last_q_val, q_val + step_width)) step_width /= 2;
                last_q_val = q_val;
                q_val += step_width;
            } else if (crt_base_size <= target_crt_base_size) {
                if (is_equal(last_q_val, q_val - step_width) ||
                    is_equal(q_val, step_width))
                    step_width /= 2;
                last_q_val = q_val;
                q_val -= step_width;
            }
            quantize(q_val);
            printf("step_width: %.15f\n", step_width);
        }
    }

    wandb_t get_q_const() {
        for (auto layer : m_layer) {
            if (layer->get_q_const() != 0) {
                return layer->get_q_const();
            }
        }
        return 0;
    }

    dim_t get_input_dims() const { return m_input_dims; }
    dim_t get_output_dims() const { return m_output_dims; }
    size_t get_input_size() const { return m_input_size; }
    size_t get_output_size() const { return m_output_size; }
    vector<Layer*> get_layer() const { return m_layer; }
    q_val_t get_min_plain_q_val() {
        q_val_t min{Q_VAL_MAX};
        for (auto l : m_layer) min = std::min(min, l->get_min_plain_q_val());
        return min;
    }
    q_val_t get_max_plain_q_val() {
        q_val_t max{Q_VAL_MIN};
        for (auto l : m_layer) max = std::max(max, l->get_max_plain_q_val());
        return max;
    }
    q_val_t get_range_plain_q_val() {
        return std::abs(get_max_plain_q_val()) +
               std::abs(get_min_plain_q_val());
    }
    wandb_t get_min_plain_val() {
        wandb_t min{WANDB_VAL_MAX};
        for (auto l : m_layer) min = std::min(min, l->get_min_plain_val());
        return min;
    }
    wandb_t get_max_plain_val() {
        wandb_t max{WANDB_VAL_MIN};
        for (auto l : m_layer) max = std::max(max, l->get_max_plain_val());
        return max;
    }
    wandb_t get_range_plain_val() {
        return std::abs(get_max_plain_val()) + std::abs(get_min_plain_val());
    }

    int get_ql() const { return m_ql; }
    std::vector<int> get_qs() const { return m_qs; }

    void print() {
        for (auto layer : m_layer) {
            layer->print();
        }
    }

   private:
    vector<ScalarTensor<q_val_t>> vec_quantize(
        vector<ScalarTensor<wandb_t>>& data, wandb_t q_const,
        int nr_samples = -1) {
        vector<ScalarTensor<q_val_t>> quantized_data;
        int cnt = nr_samples;
        if (nr_samples == -1) {
            cnt = data.size();
        }
        for (int i = 0; i < cnt; i++) {
            auto tensor = ScalarTensor<q_val_t>::quantize(
                data.at(i), QuantizationMethod::SimpleQuant, q_const);
            quantized_data.push_back(tensor);
        }

        return quantized_data;
    }

    inline bool is_equal(double x, double y) {
        if (std::fabs(x - y) < std::numeric_limits<double>::epsilon())
            return true;  // they are same
        return false;     // they are not same
    }
};

#endif
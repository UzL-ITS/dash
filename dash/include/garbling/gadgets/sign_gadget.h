#ifndef SIGN_GADGET_H
#define SIGN_GADGET_H

#include <cstdlib>
#include <cstring>
#include <vector>

#include "garbling/gadgets/approx_res_gadget.h"
#include "garbling/garbled_circuit_interface.h"
#include "garbling/gates/mixed_mod_half_gate.h"
#include "garbling/gates/projection_gate.h"
#include "misc/datatypes.h"
#include "misc/misc.h"

#ifndef SGX
#include <cuda_runtime_api.h>

#include "misc/cuda_util.h"
#endif

using std::vector;

crt_val_t mrs_add_4_functionality(crt_val_t x, void* params) {
    mrs_val_t mrs_modulus = *(mrs_val_t*)params;
    return x / mrs_modulus;
}

struct sign_functionality_params {
    mrs_val_t ms_mrs_modulus;
    int lower_bound;
    int upper_bound;
};

crt_val_t sign_functionality(crt_val_t x, void* params) {
    sign_functionality_params* p = (sign_functionality_params*)params;
    mrs_val_t cpm_value = p->ms_mrs_modulus / 2;
    return x < cpm_value ? p->upper_bound : p->lower_bound;
}

#ifndef SGX
__global__ void eval_approx_res_gadget(crt_val_t* mrs_label,
                                       crt_val_t* m_dev_in_label, int nr_inputs,
                                       __uint128_t* m_dev_approx_res_ciphers,
                                       mrs_val_t* mrs_base, int mrs_base_size,
                                       int mrs_base_nr_comps, size_t crt_base_size,
                                       crt_val_t crt_modulus, int crt_base_sum,
                                       int crt_base_prefix, size_t crt_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nr_inputs) {
        size_t nr_comps_in = get_nr_comps(crt_modulus);

        // Get projection arguments
        //// Get mrs base label
        crt_val_t* this_mrs_label = mrs_label;
        ////// Pointer for input idx
        this_mrs_label += idx * crt_base_size * mrs_base_nr_comps;
        ////// Pointer for crt_base
        this_mrs_label += crt_idx * mrs_base_nr_comps;
        //// Get cipher
        __uint128_t* cipher = m_dev_approx_res_ciphers;
        ////// Pointer for input idx
        cipher += idx * mrs_base_size * crt_base_sum;
        ////// Pointer for crt_base
        cipher += mrs_base_size * crt_base_prefix;

        for (size_t i = 0; i < mrs_base_size; ++i) {
            size_t nr_comps_out = get_nr_comps(mrs_base[i]);

            eval_proj(this_mrs_label, &m_dev_in_label[idx * nr_comps_in],
                      cipher, crt_modulus, mrs_base[i], nr_comps_in,
                      nr_comps_out);

            cipher += crt_modulus;
            this_mrs_label += nr_comps_out;
        }
    }
}


__global__ void mrs_sum_most_sig(crt_val_t* output_label, crt_val_t* mrs_label,
                                 size_t nr_inputs, mrs_val_t* dev_mrs_base,
                                 size_t mrs_base_size, int mrs_base_nr_comps,
                                 int partial_mrs_sum, size_t crt_base_size,
                                 crt_val_t** dev_dev_zero_label,
                                 __uint128_t* dev_1_cast_ciphers,
                                 __uint128_t* dev_2_cast_ciphers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nr_inputs) {
        // initialize carry
        auto least_sig_modulus = dev_mrs_base[mrs_base_size - 1];
        int nr_comps_carry = get_nr_comps(least_sig_modulus);
        crt_val_t carry[128];
        memcpy(carry, dev_dev_zero_label[least_sig_modulus],
               nr_comps_carry * sizeof(crt_val_t));

        // Get mrs label
        crt_val_t* this_mrs_label = mrs_label;
        this_mrs_label += idx * crt_base_size * mrs_base_nr_comps;
        //// the mrs-loop is inverted compared to eval_approx_res_gadget
        this_mrs_label += mrs_base_nr_comps;

        auto cipher = dev_1_cast_ciphers;
        cipher += idx * (crt_base_size + 1) * partial_mrs_sum;

        auto cipher2 = dev_2_cast_ciphers;
        cipher2 += idx * (crt_base_size + 1) * partial_mrs_sum;

        auto most_sig_modulus = dev_mrs_base[0];
        auto most_sig_nr_comps = get_nr_comps(most_sig_modulus);

        crt_val_t sum[128];
        crt_val_t sum2[128];
        for (int i = mrs_base_size - 1; i >= 0; --i) {
            mrs_val_t mrs_modulus = dev_mrs_base[i];

            // Step 2.1 Add values
            int nr_comps = get_nr_comps(mrs_modulus);
            //// the mrs-loop is inverted compared to
            //// eval_approx_res_gadget
            this_mrs_label -= nr_comps;

            //// initialize mrs sum with carry
            for (int j = 0; j < nr_comps; ++j) sum[j] = carry[j];

            auto tmp = this_mrs_label;
            for (size_t j = 0; j < crt_base_size; ++j) {
                for (int k = 0; k < nr_comps; ++k) {
                    sum[k] = modulo(sum[k] + tmp[k], mrs_modulus);
                }
                tmp += mrs_base_nr_comps;
            }

            //// We do not need to compute the last carry value
            if (i == 0) break;

            // Step 2.2: Cast Values
            //// cast values
            auto in_label = this_mrs_label;
            mrs_val_t out_modulus = (crt_base_size + 1) * mrs_modulus;
            int nr_comps_out = get_nr_comps(out_modulus);
            int size = (crt_base_size + 1) * nr_comps_out;

            crt_val_t* out_label = new crt_val_t[size];
            crt_val_t* this_out_label = out_label;

            for (size_t j = 0; j < crt_base_size; ++j) {
                eval_proj(this_out_label, in_label, cipher, mrs_modulus,
                          out_modulus, nr_comps, nr_comps_out);

                this_out_label += nr_comps_out;
                in_label += mrs_base_nr_comps;
                cipher += mrs_modulus;
            }
            //// cast carry
            eval_proj(this_out_label, carry, cipher, mrs_modulus, out_modulus,
                      nr_comps, nr_comps_out);
            cipher += mrs_modulus;

            // Step 2.3: Add casted values
            this_out_label = out_label;
            memcpy(sum2, this_out_label, nr_comps_out * sizeof(crt_val_t));

            for (size_t j = 1; j < crt_base_size + 1; ++j) {
                this_out_label += nr_comps_out;
                for (int k = 0; k < nr_comps_out; ++k) {
                    sum2[k] = modulo(sum2[k] + this_out_label[k], out_modulus);
                }
            }

            // Step 2.4: Compute carry out
            auto in_modulus2 = out_modulus;
            auto new_out_modulus = dev_mrs_base[i - 1];
            auto nr_comps_out2 = get_nr_comps(new_out_modulus);

            eval_proj(carry, sum2, cipher2, in_modulus2, new_out_modulus,
                      nr_comps_out, nr_comps_out2);

            cipher2 += in_modulus2;

            delete[] out_label;
        }
        crt_val_t* out = output_label;
        out += idx * most_sig_nr_comps;
        for (int i = 0; i < most_sig_nr_comps; ++i) {
            out[i] = sum[i];
        }
    }
}

__global__ void eval_sign(crt_val_t* output_label, crt_val_t* mrs_sum_label,
                              size_t nr_inputs, __uint128_t* m_dev_sign_ciphers,
                              size_t crt_base_size, size_t crt_base_idx,
                              crt_val_t out_modulus, size_t nr_out_moduli,
                              mrs_val_t* dev_mrs_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 3
    if (idx < nr_inputs) {
        int most_sig_nr_comps = get_nr_comps(dev_mrs_base[0]);
        int nr_comps_out = get_nr_comps(out_modulus);

        auto cipher = m_dev_sign_ciphers;
        cipher += idx * nr_out_moduli * dev_mrs_base[0] +
                  crt_base_idx * dev_mrs_base[0];
        auto sum_label = mrs_sum_label;
        sum_label += idx * most_sig_nr_comps;
        auto out = output_label + idx * nr_comps_out;
        eval_proj(out, sum_label, cipher, dev_mrs_base[0], out_modulus,
                  most_sig_nr_comps, nr_comps_out);
    }
}
#endif

class SignGadget {
    GarbledCircuitInterface* m_gc;
    // Return value for negative input
    int m_lower_bound;
    // Return value for positive input
    int m_upper_bound;
    size_t m_nr_inputs;
    size_t m_nr_outputs;
    // Output moduli of the gadget
    vector<crt_val_t> m_out_moduli;

    // Sizes of ciphertext arrays needed to copy to device
    vector<int> cipher_sizes;

    // Step 1: Approx. gadget for the first step of the approx. sign gadget
    //// size: crt base size x input size
    vector<vector<ApproxResGadget*>> m_approx_res_gadgets;
    //// size: input size x mrs base size x sum(crt base)
    __uint128_t* m_approx_res_ciphers;

    // Step 2.2: Projection gates for the second step of the mrs addition
    //// size: nr_inputs x mrs_base_size x (crt_base_size+1)
    //// (+1) for the carry value
    vector<vector<vector<ProjectionGate*>>> m_proj_gates_mrs_add_1_cast;
    //// size: nr_inputs * (crt_base_size + 1) * sum(mrs_base[1:])
    __uint128_t* m_1_cast_ciphers;

    // Step 2.4: Projection gates for the last step of the mixed radix addition
    //// size: nr_inputs * (crt_base_size + 1) * sum(mrs_base[1:])
    //// (-1), because last carry is not needed
    vector<vector<ProjectionGate*>> m_proj_gates_mrs_add_2_cast;
    __uint128_t* m_2_cast_ciphers;

    // Step 3.1: Projection gate to get sign bit
    //// size: nr_inputs x crt_base_size
    vector<vector<ProjectionGate*>> m_proj_gates_sign;
    //// size: nr_inputs x crt_base_size x mrs-base[0]
    __uint128_t* m_sign_ciphers;

    crt_val_t** m_dev_out_label{nullptr};
    crt_val_t* m_dev_mrs_label{nullptr};
    crt_val_t* m_dev_mrs_sum_most_sig_label{nullptr};

    __uint128_t* m_dev_approx_res_ciphers{nullptr};
    __uint128_t* m_dev_1_cast_ciphers{nullptr};
    __uint128_t* m_dev_2_cast_ciphers{nullptr};
    __uint128_t* m_dev_sign_ciphers{nullptr};

   public:
    SignGadget(GarbledCircuitInterface* gc, int lower_bound, int upper_bound,
               size_t nr_inputs, vector<crt_val_t> out_moduli)
        : m_gc{gc},
          m_lower_bound{lower_bound},
          m_upper_bound{upper_bound},
          m_nr_inputs{nr_inputs},
          m_nr_outputs{nr_inputs},
          m_out_moduli{out_moduli} {
        assert(m_gc->get_crt_modulus() % 2 == 0 &&
               "Approx. garbled sign computation requires an even CRT "
               "modulus");
        assert(m_gc->get_mrs_base().size() > 0 &&
               "Approx. garbled sign computation requires an MRS base with at "
               "least one element");
        assert(m_gc->get_mrs_base()[0] % 2 == 0 &&
               "Approx. garbled sign computation requires first modulus of "
               "MRS "
               "base to be even");
        assert(out_moduli.size() <= m_gc->get_crt_base().size() &&
               "At the moment our implementation of the approx. garbled sign "
               "computation requires out_moduli.size() to be "
               "less than or equal to crt_base.size()");

        // Allocate memory for approx res gadgets
        size_t size;
        size_t crt_base_size = m_gc->get_crt_base().size();
        size_t mrs_base_size = m_gc->get_mrs_base().size();
        int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                           m_gc->get_crt_base().end(), 0);

        size = nr_inputs * mrs_base_size * crt_base_sum;

        cipher_sizes.push_back(size);
        m_approx_res_ciphers = new __uint128_t[size];

        int crt_base_prefix = 0;
        for (size_t i = 0; i < crt_base_size; ++i) {
            m_approx_res_gadgets.push_back(vector<ApproxResGadget*>());
            auto in_modulus = m_gc->get_crt_base().at(i);
            for (size_t j = 0; j < nr_inputs; ++j) {
                //  Get cipher index
                //  - Index for input j
                auto ciphers = m_approx_res_ciphers;
                ciphers += j * mrs_base_size * crt_base_sum;
                // - Index for crt base i
                ciphers += mrs_base_size * crt_base_prefix;

                auto arg = new ApproxResGadget(m_gc, ciphers, in_modulus);
                m_approx_res_gadgets[i].push_back(arg);
            }
            // - Index for crt base i
            crt_base_prefix += in_modulus;
        }

        // Reserve memory for projection gates
        //// Step 2.2
        ////// (+1) Because we don't need the last carry value
        int partial_mrs_base_sum = std::accumulate(
            m_gc->get_mrs_base().begin() + 1, m_gc->get_mrs_base().end(), 0);
        ////// (+1) Because we also need to cast the carry value
        size = nr_inputs * (crt_base_size + 1) * partial_mrs_base_sum;
        cipher_sizes.push_back(size);
        m_1_cast_ciphers = new __uint128_t[size];

        //// Step 2.4
        size = nr_inputs * (mrs_base_size - 1) * (crt_base_size + 1) *
               partial_mrs_base_sum;
        cipher_sizes.push_back(size);
        m_2_cast_ciphers = new __uint128_t[size];

        //// Step 3
        auto ms_mrs_modulus = m_gc->get_mrs_base().at(0);
        size = nr_inputs * m_out_moduli.size() * ms_mrs_modulus;
        cipher_sizes.push_back(size);
        m_sign_ciphers = new __uint128_t[size];
    }

    ~SignGadget() {
        // clean up allocations on host
        for (auto& vec : m_approx_res_gadgets) {
            for (auto& gadget : vec) {
                delete gadget;
            }
        }
        for (auto& vec1 : m_proj_gates_mrs_add_1_cast) {
            for (auto& vec2 : vec1) {
                for (auto& gate : vec2) {
                    delete gate;
                }
            }
        }
        for (auto& vec : m_proj_gates_mrs_add_2_cast) {
            for (auto& gate : vec) {
                delete gate;
            }
        }
        for (auto& gates : m_proj_gates_sign) {
            for (auto& gate : gates) {
                delete gate;
            }
        }
        delete[] m_sign_ciphers;

        delete[] m_approx_res_ciphers;
        delete[] m_1_cast_ciphers;
        delete[] m_2_cast_ciphers;

#ifndef SGX
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_out_moduli.size(); ++i) {
                cudaCheckError(cudaFree(m_dev_out_label[i]));
            }
            delete[] m_dev_out_label;
        }
        if (m_dev_mrs_label != nullptr) {
            cudaCheckError(cudaFree(m_dev_mrs_label));
        }
        if (m_dev_approx_res_ciphers != nullptr) {
            cudaCheckError(cudaFree(m_dev_approx_res_ciphers));
        }
        if (m_dev_1_cast_ciphers != nullptr) {
            cudaCheckError(cudaFree(m_dev_1_cast_ciphers));
        }
        if (m_dev_2_cast_ciphers != nullptr) {
            cudaCheckError(cudaFree(m_dev_2_cast_ciphers));
        }
        if (m_dev_mrs_sum_most_sig_label != nullptr) {
            cudaCheckError(cudaFree(m_dev_mrs_sum_most_sig_label));
        }
        if (m_dev_sign_ciphers != nullptr) {
            cudaCheckError(cudaFree(m_dev_sign_ciphers));
        }
#else
        if (m_dev_out_label != nullptr) {
            for (size_t i = 0; i < m_out_moduli.size(); ++i) {
                ocall_cudaFree(m_dev_out_label[i]);
            }
            ocall_free(m_dev_out_label);
        }
        if (m_dev_mrs_label != nullptr) {
            ocall_cudaFree(m_dev_mrs_label);
        }
        if (m_dev_approx_res_ciphers != nullptr) {
            ocall_cudaFree(m_dev_approx_res_ciphers);
        }
        if (m_dev_1_cast_ciphers != nullptr) {
            ocall_cudaFree(m_dev_1_cast_ciphers);
        }
        if (m_dev_2_cast_ciphers != nullptr) {
            ocall_cudaFree(m_dev_2_cast_ciphers);
        }
        if (m_dev_mrs_sum_most_sig_label != nullptr) {
            ocall_cudaFree(m_dev_mrs_sum_most_sig_label);
        }
        if (m_dev_sign_ciphers != nullptr) {
            ocall_cudaFree(m_dev_sign_ciphers);
        }
#endif
    }

    void garble(vector<LabelTensor*>* in_label,
                vector<LabelTensor*>* out_label) {
        int crt_base_size = m_gc->get_crt_base().size();  // k
        int mrs_base_size = m_gc->get_mrs_base().size();  // n

        int mrs_base_sum_suffix = std::accumulate(
            m_gc->get_mrs_base().begin() + 1, m_gc->get_mrs_base().end(), 0);

        size_t input_size = in_label->at(0)->get_nr_label();
        m_proj_gates_mrs_add_1_cast.resize(input_size);
        m_proj_gates_mrs_add_2_cast.resize(input_size);
        m_proj_gates_sign.resize(input_size);

#ifndef SGX
#pragma omp parallel for
#endif
        for (size_t i = 0; i < input_size; ++i) {
            // cipher counter
            int cnt_1_cast = i * (crt_base_size + 1) * mrs_base_sum_suffix;
            int cnt_2_cast = i * (crt_base_size + 1) * mrs_base_sum_suffix;
            int cnt_sign = i * m_out_moduli.size() * m_gc->get_mrs_base().at(0);

            // Step 1: Approx. residues
            //// crt base size x mrs base size
            vector<vector<LabelTensor*>> mrs_label;
            for (int j = 0; j < crt_base_size; ++j) {
                auto tmp = m_approx_res_gadgets.at(j).at(i)->garble(
                    in_label->at(j)->get_label(i), j);
                mrs_label.push_back(tmp);
            }

            // Step 2: Mixed-Radix addition
            crt_val_t in_modulus = m_gc->get_mrs_base().back();
            LabelTensor carry{m_gc->get_zero_label(in_modulus)};

            vector<LabelTensor> mrs_sum;
            vector<vector<ProjectionGate*>> vec_proj_gates_cast1;
            vector<ProjectionGate*> proj_gates_cast2;

            //// From least to most significant mrs component do...
            for (int k = mrs_base_size - 1; k >= 0; --k) {
                mrs_val_t in_modulus = m_gc->get_mrs_base()[k];

                //// Step 2.1: Add values
                LabelTensor sum{carry};
                for (int j = 0; j < crt_base_size; ++j) {
                    sum += *mrs_label.at(j).at(k);
                }

                mrs_sum.push_back(sum);
                //// We do not need to compute the last carry value
                if (k == 0) break;

                //// Step 2.2: Cast values
                auto id_func = &ProjectionFunctionalities::identity;
                mrs_val_t out_modulus = (crt_base_size + 1) * in_modulus;
                vector<LabelTensor> out_base_label;
                vector<ProjectionGate*> proj_gates;
                ////// get offset label
                auto out_offset_label = m_gc->get_label_offset(out_modulus);
                ////// cast values
                for (int j = 0; j < crt_base_size; ++j) {
                    auto in_label = mrs_label.at(j).at(k)->get_label(0);
                    //////// reserve ciphers for the projection gate
                    __uint128_t* ciphers = &m_1_cast_ciphers[cnt_1_cast];
                    cnt_1_cast += in_modulus;
                    //////// reserve out_base_label
                    LabelTensor base_label{out_modulus};
                    base_label.init_random();
                    out_base_label.push_back(base_label);
                    auto proj_gate =
                        new ProjectionGate(out_modulus, ciphers, id_func, m_gc);
                    proj_gate->garble(in_label, &out_base_label.back(),
                                      out_offset_label);
                    proj_gates.push_back(proj_gate);
                }
                ////// cast carry
                //////// reserve ciphers for the projection gate
                __uint128_t* ciphers = &m_1_cast_ciphers[cnt_1_cast];
                cnt_1_cast += in_modulus;

                //////// reserve out_base_label
                LabelTensor base_label{out_modulus};
                base_label.init_random();
                out_base_label.push_back(base_label);

                auto proj_gate =
                    new ProjectionGate(out_modulus, ciphers, id_func, m_gc);
                proj_gate->garble(carry.get_label(0), &out_base_label.back(),
                                  out_offset_label);
                proj_gates.push_back(proj_gate);

                vec_proj_gates_cast1.push_back(proj_gates);

                //// Step 2.3: Add casted values
                LabelTensor sum2 = out_base_label.at(0);
                for (size_t j = 1; j < out_base_label.size(); ++j) {
                    sum2 += out_base_label.at(j);
                }

                //// Step 2.4: Compute carry out
                ////// reserve ciphers for the projection gate
                auto in_modulus2 = out_modulus;
                auto ciphers2 = &m_2_cast_ciphers[cnt_2_cast];
                cnt_2_cast += in_modulus2;

                mrs_val_t new_out_modulus = m_gc->get_mrs_base()[k - 1];
                auto proj_gate2 = new ProjectionGate(
                    new_out_modulus, ciphers2, &mrs_add_4_functionality, m_gc);
                auto new_out_offset_label =
                    m_gc->get_label_offset(new_out_modulus);
                LabelTensor new_out_base_label{new_out_modulus};
                new_out_base_label.init_random();

                proj_gate2->garble(sum2.get_label(0), &new_out_base_label,
                                   new_out_offset_label,
                                   reinterpret_cast<void*>(&in_modulus));
                proj_gates_cast2.push_back(proj_gate2);
                carry = new_out_base_label;
            }
            m_proj_gates_mrs_add_1_cast.at(i) = vec_proj_gates_cast1;
            m_proj_gates_mrs_add_2_cast.at(i) = proj_gates_cast2;

            // Step 3: Check mrs sum for sign
            vector<ProjectionGate*> proj_gates;
            auto ms_mrs_modulus = m_gc->get_mrs_base().at(0);
            sign_functionality_params params = {ms_mrs_modulus, m_lower_bound,
                                                m_upper_bound};
            for (size_t j = 0; j < m_out_moduli.size(); ++j) {
                //// Step 3.1
                __uint128_t* ciphers = &m_sign_ciphers[cnt_sign];
                auto proj_gate = new ProjectionGate(
                    m_out_moduli.at(j), ciphers, &sign_functionality, m_gc);
                auto sum_in_label{mrs_sum.back().get_label(0)};
                LabelTensor out_base_label{m_out_moduli.at(j)};  // sign label
                out_base_label.init_random();

                auto out_offset_label{
                    m_gc->get_label_offset(m_out_moduli.at(j))};
                proj_gate->garble(sum_in_label, &out_base_label,
                                  out_offset_label,
                                  reinterpret_cast<void*>(&params));
                proj_gates.push_back(proj_gate);
                cnt_sign += ms_mrs_modulus;

                out_label->at(j)->set_label(out_base_label.get_label(0), i);
            }
            m_proj_gates_sign.at(i) = proj_gates;

            // clean up
            for (auto& vec : mrs_label) {
                for (auto& label : vec) {
                    delete label;
                }
            }
        }
    }

    void cpu_evaluate(vector<LabelTensor*>* encoded_inputs,
                      vector<LabelTensor*>* out_label, int nr_threads) {
        int crt_base_size = m_gc->get_crt_base().size();  // k
        int mrs_base_size = m_gc->get_mrs_base().size();  // n

        size_t input_size = encoded_inputs->at(0)->get_nr_label();

#ifndef SGX
#pragma omp parallel for num_threads(nr_threads)
#endif
        for (size_t i = 0; i < input_size; ++i) {
            // Step 1: Approx. residues
            //// size: crt_base_size x mrs_base_size
            vector<vector<LabelTensor*>> mrs_label;
            for (int j = 0; j < crt_base_size; ++j) {
                auto tmp = m_approx_res_gadgets.at(j).at(i)->cpu_evaluate(
                    encoded_inputs->at(j)->get_label(i));
                mrs_label.push_back(tmp);
            }

            size_t crt_base_size = m_gc->get_crt_base().size();
            size_t mrs_base_size = m_gc->get_mrs_base().size();
            int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                               m_gc->get_crt_base().end(), 0);

            // Step 2: Mixed-Radix addition
            crt_val_t in_modulus = m_gc->get_mrs_base().back();
            LabelTensor carry{m_gc->get_zero_label(in_modulus)};
            vector<LabelTensor> mrs_sum;
            vector<ProjectionGate*> proj_gates;

            //// From least to most significant mrs component do...
            for (int k = mrs_base_size - 1; k >= 0; --k) {
                //// Step 2.1: Add values
                LabelTensor sum{carry};
                for (size_t j = 0; j < crt_base_size; ++j) {
                    sum += *mrs_label.at(j).at(k);
                }

                mrs_sum.push_back(sum);
                //// We do not need to compute the last carry value
                if (k == 0) break;

                //// Step 2.2: Cast values
                vector<LabelTensor> out_base_label;
                ////// cast values
                for (size_t j = 0; j < crt_base_size; ++j) {
                    auto in_label = mrs_label.at(j).at(k)->get_label(0);
                    auto cast_1_proj = m_proj_gates_mrs_add_1_cast.at(i)
                                           .at(mrs_base_size - 1 - k)
                                           .at(j);
                    auto tmp = cast_1_proj->cpu_evaluate(in_label);
                    out_base_label.push_back(tmp);
                }
                ////// cast carry
                auto cast_1_proj = m_proj_gates_mrs_add_1_cast.at(i)
                                       .at(mrs_base_size - 1 - k)
                                       .at(crt_base_size);
                auto tmp = cast_1_proj->cpu_evaluate(carry.get_label(0));
                out_base_label.push_back(tmp);

                //// Step 2.3: Add casted values
                LabelTensor sum2 = out_base_label.at(0);
                for (size_t j = 1; j < out_base_label.size(); ++j) {
                    sum2 += out_base_label.at(j);
                }

                //// Step 2.4: Compute carry out
                auto cast_2_proj =
                    m_proj_gates_mrs_add_2_cast.at(i).at(mrs_base_size - 1 - k);
                carry = cast_2_proj->cpu_evaluate(sum2.get_label(0));
            }

            // Step 3: Check mrs sum for sign
            for (size_t j = 0; j < m_out_moduli.size(); ++j) {
                auto sign_label = m_proj_gates_sign.at(i).at(j)->cpu_evaluate(
                    mrs_sum.back().get_label(0));

                out_label->at(j)->set_label(sign_label.get_label(0), i);
            }

            // clean up
            for (auto& vec : mrs_label) {
                for (auto& label : vec) {
                    delete label;
                }
            }
        }
    }
#ifndef SGX
    void cuda_move() {
        // Allocate memory for the output labels
        // - Allocate pointer array
        m_dev_out_label = new crt_val_t*[m_out_moduli.size()];

        for (size_t i = 0; i < m_out_moduli.size(); ++i) {
            size_t nr_comps = LabelTensor::get_nr_comps(m_out_moduli.at(i));
            size_t size = m_nr_outputs * nr_comps * sizeof(crt_val_t);
            cudaCheckError(cudaMalloc(
                reinterpret_cast<void**>(&m_dev_out_label[i]), size));
        }

        // Allocate memory for mrs_label ins step 1
        int crt_base_size = m_gc->get_crt_base().size();
        size_t mrs_base_nr_comps = std::accumulate(
            m_gc->get_mrs_base().begin(), m_gc->get_mrs_base().end(), 0lu,
            [](size_t a, size_t b) {
                return a + LabelTensor::get_nr_comps(b);
            });
        size_t size = m_nr_inputs * crt_base_size * mrs_base_nr_comps;
        cudaCheckError(cudaMalloc(reinterpret_cast<void**>(&m_dev_mrs_label),
                                  size * sizeof(crt_val_t)));
        // Allocate memory for sign computation in Step 2 and 3
        size_t crt_base_nr_comps = std::accumulate(
            m_gc->get_crt_base().begin(), m_gc->get_crt_base().end(), 0lu,
            [](size_t a, size_t b) {
                return a + LabelTensor::get_nr_comps(b);
            });
        auto most_sig_modulus = m_gc->get_mrs_base().at(0);
        auto most_sig_nr_comps = LabelTensor::get_nr_comps(most_sig_modulus);
        size = m_nr_inputs * most_sig_nr_comps;
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_mrs_sum_most_sig_label),
                       size * sizeof(crt_val_t)));

        // Move ciphers
        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_approx_res_ciphers),
                       cipher_sizes.at(0) * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(
            m_dev_approx_res_ciphers, m_approx_res_ciphers,
            cipher_sizes.at(0) * sizeof(__uint128_t), cudaMemcpyHostToDevice));

        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_1_cast_ciphers),
                       cipher_sizes.at(1) * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_1_cast_ciphers, m_1_cast_ciphers,
                                  cipher_sizes.at(1) * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));

        cudaCheckError(
            cudaMalloc(reinterpret_cast<void**>(&m_dev_2_cast_ciphers),
                       cipher_sizes.at(2) * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_2_cast_ciphers, m_2_cast_ciphers,
                                  cipher_sizes.at(2) * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));

        cudaCheckError(cudaMalloc(reinterpret_cast<void**>(&m_dev_sign_ciphers),
                                  cipher_sizes.at(3) * sizeof(__uint128_t)));
        cudaCheckError(cudaMemcpy(m_dev_sign_ciphers, m_sign_ciphers,
                                  cipher_sizes.at(3) * sizeof(__uint128_t),
                                  cudaMemcpyHostToDevice));
    }

    void cuda_evaluate(crt_val_t** dev_in_label) {
        size_t crt_base_size = m_gc->get_crt_base().size();
        cudaStream_t stream[crt_base_size];

        size_t nr_blocks = ceil_div(m_nr_outputs, 32lu);

        // Step 1: Approx. residues
        int crt_base_sum = std::accumulate(m_gc->get_crt_base().begin(),
                                           m_gc->get_crt_base().end(), 0);
        auto dev_mrs_base = m_gc->get_dev_mrs_base();
        int mrs_base_size = m_gc->get_mrs_base().size();
        size_t mrs_base_nr_comps = std::accumulate(
            m_gc->get_mrs_base().begin(), m_gc->get_mrs_base().end(), 0lu,
            [](size_t a, size_t b) {
                return a + LabelTensor::get_nr_comps(b);
            });

        int crt_base_prefix = 0;
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t crt_modulus = m_gc->get_crt_base().at(i);
            cudaCheckError(cudaStreamCreate(&stream[i]));

            eval_approx_res_gadget<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_mrs_label, dev_in_label[i], m_nr_inputs,
                m_dev_approx_res_ciphers, dev_mrs_base, mrs_base_size,
                mrs_base_nr_comps, crt_base_size, crt_modulus, crt_base_sum,
                crt_base_prefix, i);

            crt_base_prefix += m_gc->get_crt_base().at(i);
        }
        cudaDeviceSynchronize();

        // Step 2: Mixed-radix addition
        auto dev_dev_zero_label = m_gc->get_dev_dev_zero_label();
        int partial_mrs_base_sum = std::accumulate(
            m_gc->get_mrs_base().begin() + 1, m_gc->get_mrs_base().end(), 0);

        mrs_sum_most_sig<<<nr_blocks, 32>>>(
            m_dev_mrs_sum_most_sig_label, m_dev_mrs_label, m_nr_inputs,
            dev_mrs_base, mrs_base_size, mrs_base_nr_comps,
            partial_mrs_base_sum, crt_base_size, dev_dev_zero_label,
            m_dev_1_cast_ciphers, m_dev_2_cast_ciphers);

        cudaDeviceSynchronize();

        // Step 3: Check mrs sum for sign
        for (size_t i = 0; i < m_out_moduli.size(); ++i) {
            crt_val_t out_modulus = m_out_moduli.at(i);
            eval_sign<<<nr_blocks, 32, 0, stream[i]>>>(
                m_dev_out_label[i], m_dev_mrs_sum_most_sig_label, m_nr_inputs,
                m_dev_sign_ciphers, crt_base_size, i, out_modulus,
                m_out_moduli.size(), dev_mrs_base);
        }

        for (size_t i = 0; i < crt_base_size; ++i) {
            cudaStreamDestroy(stream[i]);
        }
        cudaDeviceSynchronize();
    }

    void cuda_move_output(vector<LabelTensor*>* out_label) {
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < m_out_moduli.size(); ++i) {
            crt_val_t modulus = m_out_moduli.at(i);
            size_t nr_comps = LabelTensor::get_nr_comps(modulus);
            cudaCheckError(cudaMemcpy(
                out_label->at(i)->get_components(), m_dev_out_label[i],
                m_nr_outputs * nr_comps * sizeof(crt_val_t),
                cudaMemcpyDeviceToHost));
        }
    }

#else
    /**
     * @brief Allocate memory for output labels on GPU.
     *
     */
    void cuda_move() {
        // Allocate memory for the output labels
        // - Allocate pointer array
        m_dev_out_label = new crt_val_t*[m_out_moduli.size()];
        int size = m_out_moduli.size() * sizeof(crt_val_t*);
        ocall_alloc_ptr_array(reinterpret_cast<void***>(&m_dev_out_label),
                              size);

        // - Allocate device array
        for (size_t i = 0; i < m_out_moduli.size(); ++i) {
            auto modulus{m_gc->get_crt_base().at(i)};
            int nr_comps = LabelTensor::get_nr_comps(modulus);
            size_t size = m_nr_outputs * nr_comps * sizeof(crt_val_t);
            ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_out_label[i]),
                             size);
        }

        // Allocate memory for mrs_label ins step 1
        int crt_base_size = m_gc->get_crt_base().size();
        size_t mrs_base_nr_comps = std::accumulate(
            m_gc->get_mrs_base().begin(), m_gc->get_mrs_base().end(), 0lu,
            [](size_t a, size_t b) {
                return a + LabelTensor::get_nr_comps(b);
            });
        size = m_nr_inputs * crt_base_size * mrs_base_nr_comps;

        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_mrs_label),
                         size * sizeof(crt_val_t));
        // Allocate memory for sign computation in Step 2 and 3
        size_t crt_base_nr_comps = std::accumulate(
            m_gc->get_crt_base().begin(), m_gc->get_crt_base().end(), 0lu,
            [](size_t a, size_t b) {
                return a + LabelTensor::get_nr_comps(b);
            });
        auto most_sig_modulus = m_gc->get_mrs_base().at(0);
        auto most_sig_nr_comps = LabelTensor::get_nr_comps(most_sig_modulus);
        size = m_nr_inputs * most_sig_nr_comps;
        ocall_cudaMalloc(
            reinterpret_cast<void**>(&m_dev_mrs_sum_most_sig_label),
            size * sizeof(crt_val_t));

        // Move ciphers
        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_approx_res_ciphers),
                         cipher_sizes.at(0) * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_approx_res_ciphers, m_approx_res_ciphers,
                               cipher_sizes.at(0) * sizeof(__uint128_t));

        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_1_cast_ciphers),
                         cipher_sizes.at(1) * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_1_cast_ciphers, m_1_cast_ciphers,
                               cipher_sizes.at(1) * sizeof(__uint128_t));

        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_2_cast_ciphers),
                         cipher_sizes.at(2) * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_2_cast_ciphers, m_2_cast_ciphers,
                               cipher_sizes.at(2) * sizeof(__uint128_t));

        ocall_cudaMalloc(reinterpret_cast<void**>(&m_dev_sign_ciphers),
                         cipher_sizes.at(3) * sizeof(__uint128_t));
        sgx_cudaMemcpyToDevice(m_dev_sign_ciphers, m_sign_ciphers,
                               cipher_sizes.at(3) * sizeof(__uint128_t));
    }
    /**
     * @brief Evaluate garbled play layer on GPU.
     *
     * @param dev_in_label
     */
    void cuda_evaluate(crt_val_t** dev_in_label) {
        int crt_base_size = m_gc->get_crt_base().size();
        crt_val_t* crt_base = m_gc->get_crt_base().data();
        int crt_base_sum =
            std::accumulate(crt_base, crt_base + crt_base_size, 0,
                            [](int a, int b) { return a + b; });
        int mrs_base_size = m_gc->get_mrs_base().size();

        mrs_val_t* dev_mrs_base = m_gc->get_dev_mrs_base();
        size_t mrs_base_nr_comps = std::accumulate(
            m_gc->get_mrs_base().begin(), m_gc->get_mrs_base().end(), 0lu,
            [](size_t a, size_t b) {
                return a + LabelTensor::get_nr_comps(b);
            });

        auto dev_dev_zero_label = m_gc->get_dev_dev_zero_label();
        int partial_mrs_base_sum = std::accumulate(
            m_gc->get_mrs_base().begin() + 1, m_gc->get_mrs_base().end(), 0);

        ocall_cuda_eval_sign(
            m_dev_mrs_label, dev_in_label, m_nr_inputs, m_nr_outputs,
            m_dev_approx_res_ciphers, dev_mrs_base, mrs_base_size,
            mrs_base_nr_comps, crt_base_size, crt_base_sum, crt_base,

            m_dev_mrs_sum_most_sig_label, partial_mrs_base_sum,
            dev_dev_zero_label, m_dev_1_cast_ciphers, m_dev_2_cast_ciphers,

            m_dev_out_label, m_dev_sign_ciphers,
            m_out_moduli.size(), m_out_moduli.data());
    }

    /**
     * @brief Move output of GPU-Evaluation to CPU.
     *
     */
    void cuda_move_output(vector<LabelTensor*>* out_label) {
        size_t crt_base_size = m_gc->get_crt_base().size();
        for (size_t i = 0; i < crt_base_size; ++i) {
            crt_val_t* comps = out_label->at(i)->get_components();
            size_t size = out_label->at(i)->size();
            crt_val_t* tmp;
            ocall_cudaMemcpyFromDevice(reinterpret_cast<void**>(&tmp),
                                       m_dev_out_label[i],
                                       size * sizeof(crt_val_t));
            std::memcpy(comps, tmp, size * sizeof(crt_val_t));
            ocall_free(reinterpret_cast<void*>(tmp));
        }
    }
#endif

    crt_val_t** get_dev_out_label() { return m_dev_out_label; }
};

#endif  // SIGN_GADGET_H
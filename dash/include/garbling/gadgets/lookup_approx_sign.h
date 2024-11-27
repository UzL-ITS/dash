#ifndef LOOKUP_APPROX_SIGN_H
#define LOOKUP_APPROX_SIGN_H

#include <math.h>

#include <vector>

#include "misc/datatypes.h"

using std::vector;

class LookupApproxSign {
    long m_discretization_level;
    size_t m_mrs_base_size;  // t
    size_t m_crt_base_size;
    mrs_val_t** m_lookup;

   public:
    LookupApproxSign(vector<crt_val_t> crt_base, vector<mrs_val_t> mrs_base)
        : m_discretization_level{std::accumulate(begin(mrs_base), end(mrs_base),
                                                 1, std::multiplies<long>())},
          m_mrs_base_size{mrs_base.size()},
          m_crt_base_size{crt_base.size()},
          m_lookup{gen_lookup(crt_base, mrs_base, m_discretization_level)} {}

    ~LookupApproxSign() {
        for (size_t i = 0; i < m_crt_base_size; i++) {
            delete[] m_lookup[i];
        }
        delete[] m_lookup;
    }

    /**
     * @brief Return pointer to most significant va lue of mrs approximation.
     *
     * @param crt_base_idx
     * @param residue
     * @return mrs_val_t*
     */
    mrs_val_t* get(int crt_base_idx, crt_val_t residue) {
        return &m_lookup[crt_base_idx][residue * m_mrs_base_size];
    }

   private:
    // https://rosettacode.org/wiki/Chinese_remainder_theorem#C
    inline uint64_t mul_inv(long long a, uint16_t b) {
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

    // https://stackoverflow.com/a/39314632/8538713
    inline double round_nearest_frac(double num, long parts) {
        double res = num * parts;
        res = round(res);
        return res / parts;
    }

    // input: most to least significant value
    inline void dec_to_mixed_radix(mrs_val_t* mixed_radix, q_val_t dec,
                                   vector<mrs_val_t> mrs_base) {
        for (int i = mrs_base.size() - 1; i >= 0; i--) {
            mixed_radix[i] = dec % mrs_base.at(i);
            dec = dec / mrs_base.at(i);
        }
    }

    mrs_val_t** gen_lookup(vector<crt_val_t> crt_base,
                           vector<mrs_val_t> mrs_base,
                           long discretization_level) {
        int t = mrs_base.size();
        long pk = std::accumulate(begin(crt_base), end(crt_base), 1,
                                  std::multiplies<long>());

        auto lookup = new mrs_val_t*[crt_base.size()];

        for (size_t i = 0; i < crt_base.size(); i++) {
            lookup[i] = new mrs_val_t[crt_base.at(i) * t];

            long double A = pk / crt_base.at(i);
            long double alpha = A * mul_inv(A, crt_base.at(i));

            for (crt_val_t j = 0; j < crt_base.at(i); j++) {
                long double d = alpha * j / pk;
                d = round_nearest_frac(d, discretization_level);
                d = d * discretization_level;

                dec_to_mixed_radix(&lookup[i][j * t],
                                   static_cast<uint32_t>(round(d)), mrs_base);
            }
        }

        return lookup;
    }
};

#endif
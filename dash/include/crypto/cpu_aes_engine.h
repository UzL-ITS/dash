#ifndef CPU_AES_ENGINE_H
#define CPU_AES_ENGINE_H

#include <cstdint>
#include <cstring>

#include "crypto/simple_aes_ni.h"
#include "misc/misc.h"

#define KEY_LENGTH 16  // in byte
#define ROUNDS 10

typedef struct key_schedule {
    unsigned char rk[(ROUNDS + 1) * KEY_LENGTH];
    unsigned int nr;
} key_schedule_t;

// Support 128 bit key length
class CPUAESEngine {
   private:
    // for example purpose only
    // https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197.pdf
    uint8_t m_fixed_key[32] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                               0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
    // OpenSSL uses big endian, intel uses little endian
    key_schedule_t m_key_schedule;

   public:
    CPUAESEngine() {
        AES_128_Key_Expansion(m_fixed_key, m_key_schedule.rk);
        m_key_schedule.nr = ROUNDS;
    }

    __uint128_t cipher(__uint128_t* plaintext) {
        __uint128_t ciphertext[1];
        AES_ECB_encrypt((unsigned char*)plaintext, (unsigned char*)ciphertext,
                        KEY_LENGTH, (const char*)m_key_schedule.rk, ROUNDS);

        return *ciphertext;
    }
};
#endif
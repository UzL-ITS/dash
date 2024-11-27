/**
 * C-Library for the Intel's AES New Instructions Set.
 * All functions except the AES_Dec_Key_From_Expansion function are from
 * https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-
 * instructions-set-paper.pdf.
 *
 * Authors: Shay Gueron, ...
 */
#ifndef SIMPLE_AES_NI_H
#define SIMPLE_AES_NI_H

#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

/**
 * Assist function for the key expansion.
 * @param temp1.
 * @param temp2.
 * @return temp1.
 */
//__extern_always_inline __m128i AES_128_ASSIST(__m128i temp1, __m128i temp2) {
__m128i AES_128_ASSIST(__m128i temp1, __m128i temp2) {
    __m128i temp3;
    temp2 = _mm_shuffle_epi32(temp2, 0xff);
    temp3 = _mm_slli_si128(temp1, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp1 = _mm_xor_si128(temp1, temp2);

    return temp1;
}

/**
 * Expand a 128-bit user key to 10(+ whitening key) round keys.
 * @param userkey Key used for the expansion.
 * @param key Collection of round keys.
 */
void AES_128_Key_Expansion(const unsigned char *userkey, unsigned char *key) {
    __m128i temp1, temp2;
    __m128i *Key_Schedule = (__m128i *) key;
    temp1 = _mm_loadu_si128((__m128i *) userkey);
    Key_Schedule[0] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x1);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[1] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x2);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[2] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x4);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[3] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x8);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[4] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x10);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[5] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x20);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[6] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x40);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[7] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x80);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[8] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x1b);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[9] = temp1;
    temp2 = _mm_aeskeygenassist_si128 (temp1, 0x36);
    temp1 = AES_128_ASSIST(temp1, temp2);
    Key_Schedule[10] = temp1;
}

//__extern_always_inline void KEY_256_ASSIST_1(__m128i* temp1, __m128i * temp2){
void KEY_256_ASSIST_1(__m128i* temp1, __m128i * temp2){
    __m128i temp4;
    *temp2 = _mm_shuffle_epi32(*temp2, 0xff);
    temp4 = _mm_slli_si128 (*temp1, 0x4);
    *temp1 = _mm_xor_si128 (*temp1, temp4);
    temp4 = _mm_slli_si128 (temp4, 0x4);
    *temp1 = _mm_xor_si128 (*temp1, temp4);
    temp4 = _mm_slli_si128 (temp4, 0x4);
    *temp1 = _mm_xor_si128 (*temp1, temp4);
    *temp1 = _mm_xor_si128 (*temp1, *temp2);
}

//__extern_always_inline void KEY_256_ASSIST_2(__m128i* temp1, __m128i * temp3){
void KEY_256_ASSIST_2(__m128i* temp1, __m128i * temp3){
    __m128i temp2, temp4;
    temp4 = _mm_aeskeygenassist_si128 (*temp1, 0x0);
    temp2 = _mm_shuffle_epi32(temp4, 0xaa);
    temp4 = _mm_slli_si128 (*temp3, 0x4);
    *temp3 = _mm_xor_si128 (*temp3, temp4);
    temp4 = _mm_slli_si128 (temp4, 0x4);
    *temp3 = _mm_xor_si128 (*temp3, temp4);
    temp4 = _mm_slli_si128 (temp4, 0x4);
    *temp3 = _mm_xor_si128 (*temp3, temp4);
    *temp3 = _mm_xor_si128 (*temp3, temp2);
}

/**
 * Expand a 256-bit user key to 14(+ whitening key) round keys.
 * @param userkey Key used for the expansion.
 * @param key Collection of round keys.
 */
void AES_256_Key_Expansion (const unsigned char *userkey,
                            unsigned char *key){
    __m128i temp1, temp2, temp3;
    __m128i *Key_Schedule = (__m128i*)key;
    temp1 = _mm_loadu_si128((__m128i*)userkey);
    temp3 = _mm_loadu_si128((__m128i*)(userkey + 16));
    Key_Schedule[0] = temp1;
    Key_Schedule[1] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x01);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[2] = temp1;
    KEY_256_ASSIST_2(&temp1, &temp3);
    Key_Schedule[3] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x02);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[4] = temp1;
    KEY_256_ASSIST_2(&temp1, &temp3);
    Key_Schedule[5] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x04);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[6] = temp1;
    KEY_256_ASSIST_2(&temp1, &temp3);
    Key_Schedule[7] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x08);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[8] = temp1;
    KEY_256_ASSIST_2(&temp1, &temp3);
    Key_Schedule[9] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x10);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[10] = temp1;
    KEY_256_ASSIST_2(&temp1, &temp3);
    Key_Schedule[11] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x20);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[12] = temp1;
    KEY_256_ASSIST_2(&temp1, &temp3);
    Key_Schedule[13] = temp3;
    temp2 = _mm_aeskeygenassist_si128 (temp3, 0x40);
    KEY_256_ASSIST_1(&temp1, &temp2);
    Key_Schedule[14] = temp1;
}

/**
 * Derive round-keys for the decryption in ecb and cbc mode.
 * @param dec_key Decryption key.
 * @param key_sched Collection of round keys.
 * @param rounds Number of AES rounds (10, 12 or 14).
 */
void AES_Dec_Key_From_Expansion(unsigned char *dec_key, const unsigned char *key_sched,
                                int rounds) {
    int i;
    __m128i tmp;
    _mm_storeu_si128(&((__m128i *) dec_key)[0], ((__m128i *) key_sched)[rounds]);
    for (i = 1; i < rounds; ++i) {
        tmp = _mm_loadu_si128(&((__m128i *) key_sched)[rounds - i]);
        tmp = _mm_aesimc_si128(tmp);
        _mm_storeu_si128(&((__m128i *) dec_key)[i], tmp);
    }
    _mm_storeu_si128(&((__m128i *) dec_key)[rounds], ((__m128i *) key_sched)[0]);
}

/**
 * Encrypt with AES ECB.
 * @param in Pointer to the plaintext.
 * @param out Pointer to the ciphertext.
 * @param length Text length in bytes.
 * @param key Pointer to the expanded key schedule (for encryption).
 * @param number_of_rounds Number of AES rounds (10, 12 or 14).
 */
void AES_ECB_encrypt(const unsigned char *in, unsigned char *out, unsigned long length,
                     const char *key, unsigned int number_of_rounds) {
    __m128i tmp;
    unsigned long i, j;
    if (length % 16)
        length = length / 16 + 1;
    else
        length = length / 16;

    for (i = 0; i < length; i++) {
        tmp = _mm_loadu_si128(&((__m128i *) in)[i]);
        tmp = _mm_xor_si128(tmp, ((__m128i *) key)[0]);

        for (j = 1; j < number_of_rounds; j++) {
            tmp = _mm_aesenc_si128(tmp, ((__m128i *) key)[j]);
        }
        tmp = _mm_aesenclast_si128(tmp, ((__m128i *) key)[j]);
        _mm_storeu_si128(&((__m128i *) out)[i], tmp);
    }
}

/**
 * Decrypt with AES ECB.
 * @param in Pointer to the ciphertext.
 * @param out Pointer to the plaintext.
 * @param length Text length in bytes.
 * @param key Pointer to the expanded key schedule (for decryption).
 * @param number_of_rounds Number of AES rounds (10, 12 or 14).
 */
void AES_ECB_decrypt(const unsigned char *in, unsigned char *out, unsigned long length,
                     const char *key, unsigned int number_of_rounds) {
    __m128i tmp;
    unsigned long i, j;
    if (length % 16)
        length = length / 16 + 1;
    else
        length = length / 16;

    for (i = 0; i < length; i++) {
        tmp = _mm_loadu_si128(&((__m128i *) in)[i]);
        tmp = _mm_xor_si128(tmp, ((__m128i *) key)[0]);

        for (j = 1; j < number_of_rounds; j++) {
            tmp = _mm_aesdec_si128(tmp, ((__m128i *) key)[j]);
        }
        tmp = _mm_aesdeclast_si128(tmp, ((__m128i *) key)[j]);
        _mm_storeu_si128(&((__m128i *) out)[i], tmp);
    }

}

/**
 * Encrypt with AES CBC.
 * @param in Pointer to the plaintext.
 * @param out Pointer to ciphertext.
 * @param ivec Initialization vector.
 * @param length Text length in bytes.
 * @param key Pointer to the expanded key schedule (for encryption).
 * @param number_of_rounds Number of AES rounds (10, 12 or 14).
 */
void AES_CBC_encrypt(const unsigned char *in, unsigned char *out,
                     unsigned char ivec[16], unsigned long length,
                     unsigned char *key, unsigned int number_of_rounds) {
    __m128i feedback, data;
    long unsigned i, j;
    if (length % 16)
        length = length / 16 + 1;
    else length /= 16;

    feedback = _mm_loadu_si128((__m128i *) ivec);

    for (i = 0; i < length; i++) {
        data = _mm_loadu_si128(&((__m128i *) in)[i]);
        feedback = _mm_xor_si128(data, feedback);
        feedback = _mm_xor_si128(feedback, ((__m128i *) key)[0]);

        for (j = 1; j < number_of_rounds; j++)
            feedback = _mm_aesenc_si128(feedback, ((__m128i *) key)[j]);

        feedback = _mm_aesenclast_si128(feedback, ((__m128i *) key)[j]);
        _mm_storeu_si128(&((__m128i *) out)[i], feedback);
    }
}

/**
 * Decrypt with AES CBC.
 * @param in Pointer to the ciphertext.
 * @param out Pointer to the plaintext.
 * @param ivec Initialization vector.
 * @param length Text length in bytes.
 * @param key Pointer to expanded key schedule (for decryption).
 * @param number_of_rounds Number of AES rounds (10, 12 or 14).
 */
void AES_CBC_decrypt(const unsigned char *in, unsigned char *out,
                     unsigned char ivec[16], unsigned long length,
                     unsigned char *key, unsigned int number_of_rounds) {
    __m128i data, feedback, last_in;
    unsigned long i, j;
    if (length % 16)
        length = length / 16 + 1;
    else length /= 16;

    feedback = _mm_loadu_si128((__m128i *) ivec);

    for (i = 0; i < length; i++) {
        last_in = _mm_loadu_si128(&((__m128i *) in)[i]);
        data = _mm_xor_si128(last_in, ((__m128i *) key)[0]);

        for (j = 1; j < number_of_rounds; j++) {
            data = _mm_aesdec_si128(data, ((__m128i *) key)[j]);
        }

        data = _mm_aesdeclast_si128(data, ((__m128i *) key)[j]);
        data = _mm_xor_si128(data, feedback);
        _mm_storeu_si128(&((__m128i *) out)[i], data);
        feedback = last_in;
    }
}


/**
 * Decrypt and encrypt with AES CTR.
 * @param in Pointer to the plaintext (encryption) / ciphertext (decryption).
 * @param out Pointer to the ciphertext (encryption) / plaintext (decryption).
 * @param ivec Initialization vector.
 * @param nonce Number only used once.
 * @param length Text length in bytes.
 * @param key Pointer to expanded key schedule (no decryption key schedule).
 * @param number_of_rounds Number of AES rounds (10, 12, 14).
 */
// void AES_CTR_en_decrypt(const unsigned char *in, unsigned char *out,
//                         const unsigned char *ivec, const unsigned char *nonce,
//                         unsigned long length, const unsigned char *key,
//                         unsigned int number_of_rounds) {
//     __m128i ctr_block, tmp, ONE, BSWAP_EPI64;

//     unsigned long i, j;

//     if (length % 16)
//         length = length / 16 + 1;
//     else length /= 16;

//     ONE = _mm_set_epi32(0, 1, 0, 0);
//     BSWAP_EPI64 = _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
//     ctr_block = _mm_insert_epi64(ctr_block, *(long long *) ivec, 1);
//     ctr_block = _mm_insert_epi32(ctr_block, *(long *) nonce, 1);
//     ctr_block = _mm_srli_si128(ctr_block, 4);
//     ctr_block = _mm_shuffle_epi8(ctr_block, BSWAP_EPI64);
//     ctr_block = _mm_add_epi64(ctr_block, ONE);

//     for (i = 0; i < length; i++) {
//         tmp = _mm_shuffle_epi8(ctr_block, BSWAP_EPI64);
//         ctr_block = _mm_add_epi64(ctr_block, ONE);
//         tmp = _mm_xor_si128(tmp, ((__m128i *) key)[0]);

//         for (j = 1; j < number_of_rounds; j++) {
//             tmp = _mm_aesenc_si128(tmp, ((__m128i *) key)[j]);
//         };

//         tmp = _mm_aesenclast_si128(tmp, ((__m128i *) key)[j]);
//         tmp = _mm_xor_si128(tmp, _mm_loadu_si128(&((__m128i *) in)[i]));
//         _mm_storeu_si128(&((__m128i *) out)[i], tmp);
//     }
// }

#endif
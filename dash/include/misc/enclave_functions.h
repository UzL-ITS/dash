#ifndef ENCLAVE_FUNCTIONS_H
#define ENCLAVE_FUNCTIONS_H

#include <cstdio>
#include <cstdarg>

#if defined(__cplusplus)
extern "C" {
#endif

/*
 * printf:
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
int printf(const char* fmt, ...) {
    char buf[BUFSIZ] = {'\0'};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

#if defined(__cplusplus)
}
#endif

#endif
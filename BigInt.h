#ifndef BIGINT_H
#define BIGINT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    int* digits;
    size_t size;
    bool isNegative;
} BigInt;

BigInt* BigInt_create(const char* number);
void BigInt_destroy(BigInt* bigInt);
BigInt* BigInt_add(const BigInt* a, const BigInt* b);
BigInt* BigInt_subtract(const BigInt* a, const BigInt* b);
BigInt* BigInt_negate(const BigInt* a);
char* BigInt_toString(const BigInt* bigInt);
BigInt* BigInt_abs(const BigInt* a);
bool BigInt_absLessThan(const BigInt* a, const BigInt* b);

#endif // BIGINT_H

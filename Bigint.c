#include "BigInt.h"

void BigInt_normalize(BigInt* bigInt) {
    while (bigInt->size > 1 && bigInt->digits[bigInt->size - 1] == 0) {
        bigInt->size--;
    }
    if (bigInt->size == 1 && bigInt->digits[0] == 0) {
        bigInt->isNegative = false;
    }
}

BigInt* BigInt_create(const char* number) {
    BigInt* bigInt = (BigInt*)malloc(sizeof(BigInt));
    bigInt->isNegative = false;
    bigInt->size = 0;
    bigInt->digits = NULL;

    size_t len = strlen(number);
    size_t start = 0;

    while (start < len && (number[start] == '+' || number[start] == '-')) {
        if (number[start] == '-') {
            bigInt->isNegative = !bigInt->isNegative;
        }
        start++;
    }

    if (start == len || (len - start == 1 && number[start] == '0')) {
        bigInt->isNegative = false;
        bigInt->size = 1;
        bigInt->digits = (int*)malloc(sizeof(int));
        bigInt->digits[0] = 0;
    } else {
        bigInt->size = len - start;
        bigInt->digits = (int*)malloc(bigInt->size * sizeof(int));
        for (size_t i = 0; i < bigInt->size; i++) {
            bigInt->digits[i] = number[len - 1 - i] - '0';
        }
    }
    BigInt_normalize(bigInt);
    return bigInt;
}

void BigInt_destroy(BigInt* bigInt) {
    if (bigInt) {
        free(bigInt->digits);
        free(bigInt);
    }
}

BigInt* BigInt_add(const BigInt* a, const BigInt* b) {
    if (a->isNegative && !b->isNegative) {
        BigInt* absA = BigInt_abs(a);
        BigInt* result = BigInt_subtract(b, absA);
        BigInt_destroy(absA);
        return result;
    }
    if (!a->isNegative && b->isNegative) {
        BigInt* absB = BigInt_abs(b);
        BigInt* result = BigInt_subtract(a, absB);
        BigInt_destroy(absB);
        return result;
    }

    BigInt* result = (BigInt*)malloc(sizeof(BigInt));
    result->isNegative = a->isNegative;
    result->size = (a->size > b->size ? a->size : b->size) + 1;
    result->digits = (int*)calloc(result->size, sizeof(int));

    int carry = 0;
    for (size_t i = 0; i < result->size; i++) {
        int sum = carry;
        if (i < a->size) sum += a->digits[i];
        if (i < b->size) sum += b->digits[i];
        result->digits[i] = sum % 10;
        carry = sum / 10;
    }
    BigInt_normalize(result);
    return result;
}

BigInt* BigInt_subtract(const BigInt* a, const BigInt* b) {
    if (a->isNegative && !b->isNegative) {
        BigInt* absA = BigInt_abs(a);
        BigInt* result = BigInt_add(absA, b);
        result->isNegative = true;
        BigInt_destroy(absA);
        return result;
    }
    if (!a->isNegative && b->isNegative) {
        BigInt* absB = BigInt_abs(b);
        BigInt* result = BigInt_add(a, absB);
        BigInt_destroy(absB);
        return result;
    }
    if (a->isNegative && b->isNegative) {
        BigInt* absA = BigInt_abs(a);
        BigInt* absB = BigInt_abs(b);
        BigInt* result = BigInt_subtract(absB, absA);
        BigInt_destroy(absA);
        BigInt_destroy(absB);
        return result;
    }

    if (BigInt_absLessThan(a, b)) {
        BigInt* result = BigInt_subtract(b, a);
        result->isNegative = true;
        return result;
    }

    BigInt* result = (BigInt*)malloc(sizeof(BigInt));
    result->isNegative = a->isNegative;
    result->size = a->size;
    result->digits = (int*)calloc(result->size, sizeof(int));

    int borrow = 0;
    for (size_t i = 0; i < a->size; i++) {
        int diff = a->digits[i] - borrow;
        if (i < b->size) diff -= b->digits[i];
        if (diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result->digits[i] = diff;
    }
    BigInt_normalize(result);
    return result;
}

BigInt* BigInt_negate(const BigInt* a) {
    BigInt* result = (BigInt*)malloc(sizeof(BigInt));
    result->isNegative = !a->isNegative;
    result->size = a->size;
    result->digits = (int*)malloc(result->size * sizeof(int));
    memcpy(result->digits, a->digits, result->size * sizeof(int));
    BigInt_normalize(result);
    return result;
}

char* BigInt_toString(const BigInt* bigInt) {
    size_t len = bigInt->size + (bigInt->isNegative ? 1 : 0);
    char* str = (char*)malloc((len + 1) * sizeof(char));
    size_t index = 0;
    if (bigInt->isNegative) {
        str[index++] = '-';
    }
    for (size_t i = 0; i < bigInt->size; i++) {
        str[index++] = bigInt->digits[bigInt->size - 1 - i] + '0';
    }
    str[len] = '\0';
    return str;
}

BigInt* BigInt_abs(const BigInt* a) {
    BigInt* result = (BigInt*)malloc(sizeof(BigInt));
    result->isNegative = false;
    result->size = a->size;
    result->digits = (int*)malloc(result->size * sizeof(int));
    memcpy(result->digits, a->digits, result->size * sizeof(int));
    return result;
}

bool BigInt_absLessThan(const BigInt* a, const BigInt* b) {
    if (a->size != b->size) {
        return a->size < b->size;
    }
    for (int i = a->size - 1; i >= 0; --i) {
        if (a->digits[i] != b->digits[i]) {
            return a->digits[i] < b->digits[i];
        }
    }
    return false;
}

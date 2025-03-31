
#include "BigInt.h"
#include "hash_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int count_pairs(BigInt** data, size_t data_size, const BigInt* target) {
    int count = 0;
    HashTable* value_count = hash_table_create(1000); // Create a hash table with 1000 buckets

    for (size_t j = 0; j < data_size; j++) {
        // Calculate the required value for data[j] to satisfy the equation
        BigInt* required_value = BigInt_add(data[j], target);
        char* required_value_str = BigInt_toString(required_value);

        // If the required value exists in the map, count the pairs
        count += hash_table_get(value_count, required_value_str);

        // Add the current value to the map or update its count
        char* current_value_str = BigInt_toString(data[j]);
        hash_table_increment(value_count, current_value_str);

        free(required_value_str);
        free(current_value_str);
        BigInt_destroy(required_value);
    }

    hash_table_destroy(value_count);
    return count;
}
int solve(char* file_name) {
    FILE* file = fopen(file_name, "r");

    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", file_name);
        return -1; // Return an error code if the file cannot be opened
    }

    char line[1024];
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Error reading target from file: %s\n", file_name);
        fclose(file);
        return -1;
    }
    line[strcspn(line, "\n")] = 0; // Remove newline character
    BigInt* target = BigInt_create(line);

    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Error reading number of integers from file: %s\n", file_name);
        BigInt_destroy(target);
        fclose(file);
        return -1;
    }
    int n = atoi(line); // Read the number of integers expected

    BigInt** vec = (BigInt**)malloc(n * sizeof(BigInt*));
    for (int i = 0; i < n; ++i) {
        if (fgets(line, sizeof(line), file)) {
            line[strcspn(line, "\n")] = 0; // Remove newline character
            vec[i] = BigInt_create(line); // Add the BigInt to the array
        } else {
            fprintf(stderr, "Invalid input: Not enough numbers provided in the file.\n");
            for (int j = 0; j < i; j++) {
                BigInt_destroy(vec[j]);
            }
            free(vec);
            BigInt_destroy(target);
            fclose(file);
            return -1; // Return an error code if not enough numbers are provided
        }
    }

    // Check if more integers are provided than expected
    if (fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Invalid input: More numbers provided than expected.\n");
        for (int i = 0; i < n; i++) {
            BigInt_destroy(vec[i]);
        }
        free(vec);
        BigInt_destroy(target);
        fclose(file);
        return -1; // Return an error code for more numbers than expected
    }

    int result = count_pairs(vec, n, target);

    for (int i = 0; i < n; i++) {
        BigInt_destroy(vec[i]);
    }
    free(vec);
    BigInt_destroy(target);
    fclose(file);

    return result;
}
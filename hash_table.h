#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <stdlib.h>
#include <string.h>

typedef struct HashNode {
    char* key;
    int value;
    struct HashNode* next;
} HashNode;

typedef struct {
    HashNode** buckets;
    size_t size;
} HashTable;

HashTable* hash_table_create(size_t size);
void hash_table_destroy(HashTable* table);
int hash_table_get(HashTable* table, const char* key);
void hash_table_put(HashTable* table, const char* key, int value);
void hash_table_increment(HashTable* table, const char* key);

#endif // HASH_TABLE_H
#include "hash_table.h"

static size_t hash_function(const char* key, size_t size) {
    size_t hash = 0;
    while (*key) {
        hash = (hash * 31) + *key++;
    }
    return hash % size;
}

HashTable* hash_table_create(size_t size) {
    HashTable* table = (HashTable*)malloc(sizeof(HashTable));
    table->buckets = (HashNode**)calloc(size, sizeof(HashNode*));
    table->size = size;
    return table;
}

void hash_table_destroy(HashTable* table) {
    for (size_t i = 0; i < table->size; i++) {
        HashNode* node = table->buckets[i];
        while (node) {
            HashNode* temp = node;
            node = node->next;
            free(temp->key);
            free(temp);
        }
    }
    free(table->buckets);
    free(table);
}

int hash_table_get(HashTable* table, const char* key) {
    size_t index = hash_function(key, table->size);
    HashNode* node = table->buckets[index];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            return node->value;
        }
        node = node->next;
    }
    return 0;
}

void hash_table_put(HashTable* table, const char* key, int value) {
    size_t index = hash_function(key, table->size);
    HashNode* node = table->buckets[index];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value = value;
            return;
        }
        node = node->next;
    }
    HashNode* new_node = (HashNode*)malloc(sizeof(HashNode));
    new_node->key = strdup(key);
    new_node->value = value;
    new_node->next = table->buckets[index];
    table->buckets[index] = new_node;
}

void hash_table_increment(HashTable* table, const char* key) {
    size_t index = hash_function(key, table->size);
    HashNode* node = table->buckets[index];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->value++;
            return;
        }
        node = node->next;
    }
    HashNode* new_node = (HashNode*)malloc(sizeof(HashNode));
    new_node->key = strdup(key);
    new_node->value = 1;
    new_node->next = table->buckets[index];
    table->buckets[index] = new_node;
}
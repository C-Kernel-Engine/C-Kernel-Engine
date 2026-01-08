/*
 * Trie Implementation
 *
 * High-performance trie for token lookups
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "data_structures/tries/trie.h"

/* Default max nodes for a vocabulary of ~50k tokens */
#define DEFAULT_MAX_NODES 1000000

/* Create a new trie node */
static CKTrieNode *create_node(void) {
    CKTrieNode *node = (CKTrieNode *)calloc(1, sizeof(CKTrieNode));
    if (node) {
        node->token_id = -1;
        node->is_special = false;
        node->priority = 0;
    }
    return node;
}

CKTrie *ck_trie_create(size_t max_nodes) {
    CKTrie *trie = (CKTrie *)malloc(sizeof(CKTrie));
    if (!trie) {
        return NULL;
    }

    if (max_nodes == 0) {
        max_nodes = DEFAULT_MAX_NODES;
    }

    trie->root = create_node();
    if (!trie->root) {
        free(trie);
        return NULL;
    }

    trie->max_nodes = max_nodes;
    trie->node_count = 1; /* Root node */

    return trie;
}

void ck_trie_free(CKTrie *trie) {
    if (!trie) return;

    /* Free all nodes using BFS */
    CKTrieNode **queue = (CKTrieNode **)malloc(trie->node_count * sizeof(CKTrieNode *));
    if (!queue) {
        /* Fallback to simple free */
        free(trie->root);
        free(trie);
        return;
    }

    size_t head = 0, tail = 0;
    queue[tail++] = trie->root;

    while (head < tail) {
        CKTrieNode *node = queue[head++];
        for (int i = 0; i < 256; i++) {
            if (node->children[i]) {
                queue[tail++] = node->children[i];
            }
        }
        free(node);
    }

    free(queue);
    free(trie);
}

void ck_trie_clear(CKTrie *trie) {
    if (!trie) return;

    /* Free all nodes except root */
    CKTrieNode **queue = (CKTrieNode **)malloc(trie->node_count * sizeof(CKTrieNode *));
    if (!queue) return;

    size_t head = 0, tail = 0;
    queue[tail++] = trie->root;

    while (head < tail) {
        CKTrieNode *node = queue[head++];
        for (int i = 0; i < 256; i++) {
            if (node->children[i]) {
                queue[tail++] = node->children[i];
            }
        }
        /* Clear children pointers but don't free root */
        if (node != trie->root) {
            free(node);
        } else {
            memset(node->children, 0, sizeof(node->children));
            node->token_id = -1;
        }
    }

    free(queue);
    trie->node_count = 1;
}

int ck_trie_insert(CKTrie *trie, const char *token, int32_t token_id, bool is_special, int32_t priority) {
    if (!trie || !token) return -1;

    CKTrieNode *node = trie->root;
    const unsigned char *p = (const unsigned char *)token;

    while (*p) {
        unsigned char c = *p++;

        if (!node->children[c]) {
            if (trie->node_count >= trie->max_nodes) {
                return -1; /* Out of nodes */
            }

            CKTrieNode *new_node = create_node();
            if (!new_node) return -1;

            node->children[c] = new_node;
            trie->node_count++;
        }

        node = node->children[c];
    }

    /* Mark end of token */
    node->token_id = token_id;
    node->is_special = is_special;
    node->priority = priority;

    return 0;
}

int32_t ck_trie_find_longest(const CKTrie *trie, const char *text, size_t text_len,
                              size_t start_pos, size_t *match_len) {
    if (!trie || !text || start_pos >= text_len || !match_len) {
        *match_len = 0;
        return -1;
    }

    CKTrieNode *node = trie->root;
    CKTrieNode *last_token_node = NULL;
    size_t last_token_len = 0;
    size_t pos = start_pos;

    /* Traverse as long as we have matching children */
    while (pos < text_len && node) {
        unsigned char c = (unsigned char)text[pos];

        if (!node->children[c]) {
            break;
        }

        node = node->children[c];
        pos++;

        /* Record if this is a valid token end */
        if (node->token_id >= 0) {
            last_token_node = node;
            last_token_len = pos - start_pos;
        }
    }

    *match_len = last_token_len;

    if (last_token_node) {
        return last_token_node->token_id;
    }

    return -1;
}

bool ck_trie_has_prefix(const CKTrie *trie, const char *text, size_t text_len, size_t pos) {
    if (!trie || !text || pos >= text_len) return false;

    CKTrieNode *node = trie->root;

    while (pos < text_len && node) {
        unsigned char c = (unsigned char)text[pos];
        node = node->children[c];
        pos++;
    }

    return node != NULL;
}

size_t ck_trie_node_count(const CKTrie *trie) {
    return trie ? trie->node_count : 0;
}

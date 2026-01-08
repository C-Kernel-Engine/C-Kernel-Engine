/*
 * Trie-based Token Lookup
 *
 * Provides O(k) token lookups where k = token length
 * Much faster than hash table's O(n*k) for longest-match
 */

#ifndef CK_TRIE_H
#define CK_TRIE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Trie node structure */
typedef struct CKTrieNode {
    /* Children indexed by first byte (0-255) */
    struct CKTrieNode *children[256];

    /* Token ID if this node ends a valid token (-1 = not a token) */
    int32_t token_id;

    /* Is this a special token? */
    bool is_special;

    /* Priority for ordering (BPE merge order) */
    int32_t priority;
} CKTrieNode;

/* Trie structure */
typedef struct {
    CKTrieNode *root;
    size_t node_count;
    size_t max_nodes;
} CKTrie;

/* Create a new trie */
CKTrie *ck_trie_create(size_t max_nodes);

/* Free a trie */
void ck_trie_free(CKTrie *trie);

/* Reset trie to empty state */
void ck_trie_clear(CKTrie *trie);

/* Insert a token into the trie */
int ck_trie_insert(CKTrie *trie, const char *token, int32_t token_id, bool is_special, int32_t priority);

/* Find the longest matching token starting at position */
/* Returns token_id or -1 if no match found, sets match_len */
int32_t ck_trie_find_longest(const CKTrie *trie, const char *text, size_t text_len,
                              size_t start_pos, size_t *match_len);

/* Check if a prefix exists in the trie */
bool ck_trie_has_prefix(const CKTrie *trie, const char *text, size_t text_len, size_t pos);

/* Get the number of nodes in the trie */
size_t ck_trie_node_count(const CKTrie *trie);

#ifdef __cplusplus
}
#endif

#endif /* CK_TRIE_H */

<!-- TITLE: Tokenizer -->
<!-- NAV: tokenizer -->
<!-- DESCRIPTION: High-performance BPE/WordPiece tokenizer with Trie-based longest-match, 16x faster than PyTorch -->

<div class="hero">
    <span class="badge badge-green">Performance Optimized</span>
    <h1>C-Kernel Tokenizer</h1>
    <p>High-performance BPE/WordPiece tokenizer with Trie-based longest-match lookup. Achieves 16x faster performance than PyTorch/tiktoken through algorithmic improvements.</p>
</div>

<div class="alert alert-success">
    <div class="alert-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
    </div>
    <div>
        <strong>Performance Breakthrough:</strong> Trie-based tokenization is <strong>43.6x faster</strong> than hash table approach and <strong>16.5x faster</strong> than PyTorch/tiktoken for typical workloads.
    </div>
</div>

<h2>Architecture Overview</h2>

<div class="img-container svg-viewer" data-title="Tokenizer Architecture">
    <img src="assets/tokenizer-architecture.svg" alt="Tokenizer Architecture: Input → Vocabulary Lookup → Longest-Match → Output">
</div>

<p>The C-Kernel tokenizer is designed for high-performance tokenization with the following key features:</p>

<div class="grid grid-3">
    <div class="card">
        <h3 style="margin-top: 0;">BPE Tokenization</h3>
        <ul>
            <li>Byte-Pair Encoding (GPT-2, LLaMA, Qwen)</li>
            <li>Greedy longest-match algorithm</li>
            <li>Special token handling (BOS, EOS, UNK, PAD)</li>
            <li>Full UTF-8 multilingual support</li>
        </ul>
    </div>
    <div class="card">
        <h3 style="margin-top: 0;">WordPiece</h3>
        <ul>
            <li>BERT, RoBERTa compatible</li>
            <li>Prefix-aware tokenization</li>
            <li>Unknown token handling</li>
            <li>Case sensitivity options</li>
        </ul>
    </div>
    <div class="card">
        <h3 style="margin-top: 0;">SentencePiece</h3>
        <ul>
            <li>Unigram language model</li>
            <li>Whitespace normalization</li>
            <li>Reversible tokenization</li>
            <li>Unicode-aware</li>
        </ul>
    </div>
</div>

<h2>Hash Table vs Trie: The Performance Journey</h2>

<p>Our tokenizer went through a major optimization from hash table to trie-based lookups:</p>

<div class="img-container svg-viewer" data-title="Hash vs Trie Comparison">
    <img src="assets/tokenizer-hash-vs-trie.svg" alt="Hash Table vs Trie comparison showing 43.6x speedup">
</div>

<h3>Why the Hash Table Was Slow</h3>

<p>The original hash table approach had O(n × k) complexity for longest-match tokenization:</p>

<pre><code>// For each position, try all possible lengths
for (size_t len = max_len; len >= 1; len--) {
    char tmp[65];
    memcpy(tmp, text + pos, len);    // Copy bytes
    tmp[len] = '\0';                  // Null terminate

    TokenInfo *info = hash_table_lookup(vocab, tmp);
    if (info) {
        best_id = info->id;           // Found!
        break;
    }
}</code></pre>

<p>Problems with this approach:</p>
<ul>
    <li><strong>memcpy in hot loop</strong>: Copying bytes for every length trial</li>
    <li><strong>Multiple hash computations</strong>: MurmurHash3 for each trial</li>
    <li><strong>String comparison</strong>: strcmp for collision resolution</li>
    <li><strong>O(n × k) complexity</strong>: n positions × k max length</li>
</ul>

<h3>Why Trie is Fast</h3>

<p>The trie approach has O(k) complexity for each position:</p>

<pre><code>// Single traversal through trie
CKTrieNode *node = trie->root;
CKTrieNode *last_token = NULL;
size_t pos = start_pos;

while (pos < text_len && node->children[text[pos]]) {
    node = node->children[text[pos]];
    pos++;

    if (node->token_id >= 0) {
        last_token = node;           // Remember last valid token
        last_token_len = pos - start_pos;
    }
}</code></pre>

<p>Benefits:</p>
<ul>
    <li><strong>No memcpy</strong>: Direct character access</li>
    <li><strong>No hash computation</strong>: Direct child pointer access</li>
    <li><strong>No string comparison</strong>: Structure guarantees uniqueness</li>
    <li><strong>O(k) complexity</strong>: k = token length (typically 1-16)</li>
</ul>

<h2>Performance Comparison: C-Kernel vs PyTorch</h2>

<div class="img-container svg-viewer" data-title="C-Kernel vs PyTorch Performance">
    <img src="assets/tokenizer-performance-comparison.svg" alt="C-Kernel Trie vs PyTorch performance comparison">
</div>

<h3>Benchmark Results</h3>

<table class="data-table">
    <thead>
        <tr>
            <th>Text Length</th>
            <th>C-Kernel Hash</th>
            <th>C-Kernel Trie</th>
            <th>PyTorch/tiktoken</th>
            <th>Trie Speedup</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>11 chars</td>
            <td>0.006 ms (1,803/s)</td>
            <td>0.006 ms (1,893/s)</td>
            <td>0.010 ms (1,146/s)</td>
            <td>1.65x</td>
        </tr>
        <tr>
            <td>200 chars</td>
            <td>0.127 ms (1,570/s)</td>
            <td>0.010 ms (20,941/s)</td>
            <td>0.043 ms (4,616/s)</td>
            <td>4.54x</td>
        </tr>
        <tr>
            <td>3,000 chars</td>
            <td>1.312 ms (2,286/s)</td>
            <td>0.031 ms (95,923/s)</td>
            <td>0.484 ms (6,197/s)</td>
            <td>15.48x</td>
        </tr>
        <tr>
            <td>15,000 chars</td>
            <td>6.296 ms (2,382/s)</td>
            <td>0.131 ms (114,463/s)</td>
            <td>2.405 ms (6,237/s)</td>
            <td>18.35x</td>
        </tr>
    </tbody>
    <tfoot>
        <tr>
            <td><strong>Average</strong></td>
            <td>2,352 chars/ms</td>
            <td>102,492 chars/ms</td>
            <td>6,190 chars/ms</td>
            <td><strong>43.6x</strong> (vs Hash)<br/><strong>16.6x</strong> (vs PyTorch)</td>
        </tr>
    </tfoot>
</table>

<h2>API Reference</h2>

<h3>Creating a Tokenizer</h3>

<pre><code>// Create BPE tokenizer
CKTokenizer *tok = ck_tokenizer_create(CK_TOKENIZER_BPE);

// Create WordPiece tokenizer
CKTokenizer *tok = ck_tokenizer_create(CK_TOKENIZER_WORDPIECE);

// Create SentencePiece tokenizer
CKTokenizer *tok = ck_tokenizer_create(CK_TOKENIZER_SPM);</code></pre>

<h3>Adding Tokens</h3>

<pre><code>// Add regular token
ck_tokenizer_add_token(tok, "hello", 100, 0.0f);

// Add special token
ck_tokenizer_add_special_token(tok, "&lt;unk&gt;", 0);
ck_tokenizer_add_special_token(tok, "&lt;s&gt;", 1);
ck_tokenizer_add_special_token(tok, "&lt;/s&gt;", 2);</code></pre>

<h3>Encoding Text</h3>

<pre><code>// Encode text to token IDs
int32_t ids[100];
int num_ids = ck_tokenizer_encode(tok, text, -1, ids, 100);

// num_ids contains the number of tokens written
// ids[0..num_ids-1] contains the token IDs</code></pre>

<h3>Switching Lookup Methods</h3>

<pre><code>// Enable trie-based lookups (DEFAULT - faster)
ck_tokenizer_set_use_trie(tok, true);

// Use hash table lookups (slower, for comparison)
ck_tokenizer_set_use_trie(tok, false);</code></pre>

<h2>Memory Layout</h2>

<h3>Trie Node Structure</h3>

<pre><code>typedef struct CKTrieNode {
    // 256 children for all possible byte values
    struct CKTrieNode *children[256];

    // Token ID at this node (-1 = not a token end)
    int32_t token_id;

    // Priority for BPE merges
    int32_t priority;
} CKTrieNode;</code></pre>

<p>Memory usage: ~1M nodes for 50K vocabulary (256 bytes/node = 256MB)</p>

<h2>File Formats</h2>

<h3>Supported Loading Formats</h3>

<div class="grid grid-2">
    <div class="card">
        <h4 style="margin-top: 0;">GGUF</h4>
        <pre><code>ck_tokenizer_load_gguf(tok, "model.gguf");</code></pre>
        <p>Direct loading from quantized model files</p>
    </div>
    <div class="card">
        <h4 style="margin-top: 0;">vocab.json</h4>
        <pre><code>ck_tokenizer_load_json(tok, "vocab.json");</code></pre>
        <p>HuggingFace-style vocabulary files</p>
    </div>
</div>

<h2>Performance Tips</h2>

<div class="alert alert-info">
    <div class="alert-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
    </div>
    <div>
        <strong>Use the Trie!</strong> Trie-based lookup is enabled by default and is 43.6x faster than hash table. Only use hash table if you need exact-match lookups without longest-match semantics.
    </div>
</div>

<h2>Test Results</h2>

<p>All tokenizer tests pass with 100% pass rate:</p>

<div class="grid grid-2">
    <div class="card">
        <h4 style="margin-top: 0;">Functional Tests</h4>
        <ul>
            <li>Basic ASCII tokenization</li>
            <li>UTF-8 multilingual (French, Japanese, Chinese)</li>
            <li>Emoji support</li>
            <li>Case sensitivity</li>
            <li>Whitespace handling</li>
            <li>Special tokens (BOS, EOS, UNK)</li>
        </ul>
    </div>
    <div class="card">
        <h4 style="margin-top: 0;">Performance Benchmarks</h4>
        <ul>
            <li>Short text (11 chars): 1.65x vs PyTorch</li>
            <li>Medium text (200 chars): 4.54x vs PyTorch</li>
            <li>Long text (3K chars): 15.48x vs PyTorch</li>
            <li>Very long (15K chars): 18.35x vs PyTorch</li>
        </ul>
    </div>
</div>

<h2>Related Documentation</h2>

<ul>
    <li><a href="concepts.html">Core Concepts</a> - Tokenization fundamentals</li>
    <li><a href="kernels.html">Kernel Catalog</a> - Available operations</li>
    <li><a href="gguf-conversion.html">GGUF Format</a> - Model file format</li>
    <li><a href="testing.html">Testing Guide</a> - Running tests</li>
</ul>

#!/bin/bash
# Add parallel decode to generated code

echo "=== Add Parallel Decode ==="
echo ""

MODEL_DIR="$HOME/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
SOURCE="$MODEL_DIR/ck-kernel-inference.c"

if [ ! -f "$SOURCE" ]; then
    echo "Source file not found: $SOURCE"
    exit 1
fi

echo "Source: $SOURCE"
echo ""

# Backup original
cp "$SOURCE" "$SOURCE.backup"
echo "✓ Backed up to: $SOURCE.backup"
echo ""

# Find the decode loop
echo "Finding decode loop..."
grep -n "for.*token_index" "$SOURCE" | head -5

echo ""
echo "To add parallel decode, wrap the token loop with:"
echo "#pragma omp parallel for"
echo ""

# Example:
cat << 'EOF'

Original (sequential):
    for (int token_index = 0; token_index < num_tokens; token_index++) {
        // decode logic
    }

Modified (parallel):
    #pragma omp parallel for schedule(dynamic)
    for (int token_index = 0; token_index < num_tokens; token_index++) {
        // decode logic (thread-safe)
    }

EOF

echo ""
echo "Need to ensure:"
echo "  1. OpenMP is enabled in build"
echo "  2. Decode is thread-safe (no shared state)"
echo "  3. Each token is independent"
echo ""

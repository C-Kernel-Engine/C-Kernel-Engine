#!/bin/bash
# Generate folder structure HTML from tree command
# This is called by build.sh to dynamically update the project structure

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_FILE="$(dirname "$0")/../_partials/folder_structure.html"

# Run a focused tree for core source areas only. Minimal servers may not have
# the optional `tree` package installed, so keep a Python fallback.
if command -v tree >/dev/null 2>&1; then
    TREE_OUTPUT=$(cd "$PROJECT_ROOT" && tree -d -L 3 --dirsfirst \
        src/kernels version/v6.6 version/v7 \
        -I '__pycache__|*.o|*.pyc|doxygen_output|build|.git|node_modules|*.bin|*.so' \
        --charset=ascii 2>/dev/null)
else
    TREE_OUTPUT=$(cd "$PROJECT_ROOT" && python3 - <<'PY'
from pathlib import Path

roots = [Path("src/kernels"), Path("version/v6.6"), Path("version/v7")]
ignore = {"__pycache__", "doxygen_output", "build", ".git", "node_modules"}
max_depth = 3
dir_count = 0

def children(path):
    try:
        items = [p for p in path.iterdir() if p.is_dir() and p.name not in ignore]
    except OSError:
        return []
    return sorted(items, key=lambda p: p.name.lower())

def emit(path, prefix, depth):
    global dir_count
    kids = children(path) if depth < max_depth else []
    for idx, child in enumerate(kids):
        last = idx == len(kids) - 1
        connector = "`-- " if last else "|-- "
        print(f"{prefix}{connector}{child.name}")
        dir_count += 1
        emit(child, prefix + ("    " if last else "|   "), depth + 1)

for root in roots:
    if not root.exists():
        continue
    print(root.as_posix())
    dir_count += 1
    emit(root, "", 1)

print()
print(f"{dir_count} directories")
PY
)
fi

# Generate HTML partial
cat > "$OUTPUT_FILE" << 'HEADER'
<div class="folder-structure">
    <div class="folder-header">
        <span class="folder-title">Focused Source Tree</span>
        <span class="folder-scope">src/kernels · version/v6.6 · version/v7</span>
        <span class="folder-updated">Updated: TIMESTAMP</span>
    </div>
    <pre class="tree-output">
HEADER

# Add the tree output (escape HTML entities)
echo "$TREE_OUTPUT" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'FOOTER'
</pre>
</div>

<style>
.folder-structure {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
    margin: 1.5rem 0;
}
.folder-header {
    background: #252525;
    padding: 12px 16px;
    display: flex;
    justify-content: flex-start;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    border-bottom: 1px solid #333;
}
.folder-title {
    color: #ffb400;
    font-weight: 600;
    font-size: 14px;
}
.folder-scope {
    color: #89d3ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.folder-updated {
    color: #666;
    font-size: 11px;
    margin-left: auto;
}
.tree-output {
    margin: 0;
    padding: 16px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.5;
    color: #b0b0b0;
    overflow-x: auto;
    white-space: pre;
}
.tree-output .directory {
    color: #6ab0f3;
}
</style>
FOOTER

# Replace timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
sed -i "s/TIMESTAMP/$TIMESTAMP/" "$OUTPUT_FILE"

echo "  Generated folder structure: $OUTPUT_FILE"

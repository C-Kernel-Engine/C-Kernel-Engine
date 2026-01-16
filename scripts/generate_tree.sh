#!/bin/bash
# Generate folder structure HTML from tree command
# This is called by build.sh to dynamically update the project structure

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUTPUT_FILE="$(dirname "$0")/../_partials/folder_structure.html"

# Run tree with nice formatting, excluding build artifacts
TREE_OUTPUT=$(cd "$PROJECT_ROOT" && tree -L 3 --dirsfirst \
    -I '__pycache__|*.o|*.pyc|doxygen_output|build|.git|node_modules|*.bin|*.so' \
    --charset=ascii 2>/dev/null)

# Generate HTML partial
cat > "$OUTPUT_FILE" << 'HEADER'
<div class="folder-structure">
    <div class="folder-header">
        <span class="folder-title">Project Structure</span>
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
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #333;
}
.folder-title {
    color: #ffb400;
    font-weight: 600;
    font-size: 14px;
}
.folder-updated {
    color: #666;
    font-size: 11px;
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

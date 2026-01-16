#!/bin/bash
# Build script for C-Kernel-Engine documentation pages
# Combines header + page content + footer into final HTML files
# Also integrates Doxygen-generated API documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTIALS_DIR="$SCRIPT_DIR/_partials"
PAGES_DIR="$SCRIPT_DIR/_pages"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"

# Get current date/time dynamically from system
CURRENT_YEAR=$(date +%Y)
CURRENT_MONTH=$(date +%Y-%m)
CURRENT_DATE=$(date +%Y-%m-%d)

echo "Building C-Kernel-Engine documentation..."
echo "  Date: $CURRENT_DATE"

# Function to inject content at a placeholder
inject_content() {
    local output_file="$1"
    local placeholder="$2"
    local content_file="$3"

    if grep -q "$placeholder" "$output_file"; then
        tmp_before=$(mktemp)
        tmp_after=$(mktemp)

        # Split file at placeholder
        sed -n "1,/$placeholder/p" "$output_file" | sed '$d' > "$tmp_before"
        sed -n "/$placeholder/,\$p" "$output_file" | sed '1d' > "$tmp_after"

        # Reassemble with injected content
        cat "$tmp_before" > "$output_file"
        cat "$content_file" >> "$output_file"
        cat "$tmp_after" >> "$output_file"

        rm -f "$tmp_before" "$tmp_after"
    fi
}

# Step 1: Run Doxygen if Doxyfile exists
if [ -f "$DOCS_DIR/Doxyfile" ]; then
    echo "  Running Doxygen..."
    (cd "$DOCS_DIR" && doxygen Doxyfile 2>/dev/null) || echo "  Warning: Doxygen had warnings, continuing"
fi

# Step 2: Generate API content from Doxygen XML
if [ -f "$SCRIPT_DIR/scripts/parse_doxygen.py" ] && [ -d "$DOCS_DIR/doxygen_output/xml" ]; then
    echo "  Parsing Doxygen XML..."
    python3 "$SCRIPT_DIR/scripts/parse_doxygen.py" || echo "  Warning: API parsing failed"
fi

# Step 3: Generate folder structure
if [ -f "$SCRIPT_DIR/scripts/generate_tree.sh" ]; then
    bash "$SCRIPT_DIR/scripts/generate_tree.sh"
fi

# Process each page in _pages directory
for page in "$PAGES_DIR"/*.html; do
    if [ -f "$page" ]; then
        filename=$(basename "$page")

        # Skip files starting with underscore (templates, partials)
        if [[ "$filename" == _* ]]; then
            echo "  Skipping template: $filename"
            continue
        fi
        pagename="${filename%.html}"

        echo "  Building: $filename"

        # Extract page metadata from comments at top of file
        # Format: <!-- TITLE: Page Title -->
        # Format: <!-- NAV: index -->
        page_title=$(grep -oP '<!--\s*TITLE:\s*\K[^-]+' "$page" | tr -d ' ' || echo "Documentation")
        nav_active=$(grep -oP '<!--\s*NAV:\s*\K\w+' "$page" || echo "")

        # Read partials
        header=$(cat "$PARTIALS_DIR/header.html")
        footer=$(cat "$PARTIALS_DIR/footer.html")

        # Read page content (skip metadata comments)
        content=$(sed '/^<!--.*-->$/d' "$page")

        # Replace template variables in header
        header="${header//\{\{PAGE_TITLE\}\}/$page_title}"

        # Clear all nav active states
        header="${header//\{\{NAV_INDEX\}\}/}"
        header="${header//\{\{NAV_QUICKSTART\}\}/}"
        header="${header//\{\{NAV_DEVGUIDE\}\}/}"
        header="${header//\{\{NAV_SCALING\}\}/}"
        header="${header//\{\{NAV_ARCHITECTURE\}\}/}"
        header="${header//\{\{NAV_KERNELS\}\}/}"
        header="${header//\{\{NAV_GEMM\}\}/}"
        header="${header//\{\{NAV_QUANTIZATION\}\}/}"
        header="${header//\{\{NAV_SIMD\}\}/}"
        header="${header//\{\{NAV_CODEGEN\}\}/}"
        header="${header//\{\{NAV_MEMORY\}\}/}"
        header="${header//\{\{NAV_PROFILING\}\}/}"
        header="${header//\{\{NAV_CONCEPTS\}\}/}"
        header="${header//\{\{NAV_TESTING\}\}/}"
        header="${header//\{\{NAV_PARITY\}\}/}"
        header="${header//\{\{NAV_NIGHTLY\}\}/}"
        header="${header//\{\{NAV_RESEARCH\}\}/}"
        header="${header//\{\{NAV_DECISIONS\}\}/}"
        header="${header//\{\{NAV_API\}\}/}"
        header="${header//\{\{NAV_CONTRIBUTING\}\}/}"

        if [ -n "$nav_active" ]; then
            header="${header//\{\{NAV_${nav_active^^}\}\}/active}"
        fi

        # Replace date variables in footer
        footer="${footer//\{\{YEAR\}\}/$CURRENT_YEAR}"
        footer="${footer//\{\{CURRENT_DATE\}\}/$CURRENT_DATE}"

        # Replace date variables in content
        content="${content//\{\{YEAR\}\}/$CURRENT_YEAR}"
        content="${content//\{\{CURRENT_MONTH\}\}/$CURRENT_MONTH}"
        content="${content//\{\{CURRENT_DATE\}\}/$CURRENT_DATE}"

        # Combine header + content + footer
        output_file="$SCRIPT_DIR/$filename"
        echo "$header" > "$output_file"
        echo "$content" >> "$output_file"
        echo "$footer" >> "$output_file"

        # Special handling for API page: inject Doxygen-generated content
        if [[ "$filename" == "api.html" ]] && [ -f "$PARTIALS_DIR/api_content.html" ]; then
            echo "    Injecting API content..."
            inject_content "$output_file" "{{API_CONTENT}}" "$PARTIALS_DIR/api_content.html"
        fi

        # Special handling for PyTorch Parity page: inject test results
        if [[ "$filename" == "pytorch-parity.html" ]] && [ -f "$PARTIALS_DIR/test_results.html" ]; then
            echo "    Injecting test results..."
            inject_content "$output_file" "{{TEST_RESULTS}}" "$PARTIALS_DIR/test_results.html"
        fi

        # Inject folder structure where placeholder exists
        if grep -q "{{FOLDER_STRUCTURE}}" "$output_file" && [ -f "$PARTIALS_DIR/folder_structure.html" ]; then
            echo "    Injecting folder structure..."
            inject_content "$output_file" "{{FOLDER_STRUCTURE}}" "$PARTIALS_DIR/folder_structure.html"
        fi
    fi
done

echo "Build complete! Generated files:"
ls -la "$SCRIPT_DIR"/*.html 2>/dev/null || echo "  No HTML files generated"

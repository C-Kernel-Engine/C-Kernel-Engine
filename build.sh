#!/bin/bash
# Build script for C-Kernel-Engine documentation pages
# Combines header + page content + footer into final HTML files
# Also integrates Doxygen-generated API documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTIALS_DIR="$SCRIPT_DIR/_partials"
PAGES_DIR="$SCRIPT_DIR/_pages"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
SITE_URL="${SITE_URL:-https://antshiv.github.io/C-Kernel-Engine}"
PAGE_METADATA_FILE="$SCRIPT_DIR/page_metadata.json"

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

build_canonical_url() {
    local filename="$1"
    if [ "$filename" = "index.html" ]; then
        printf '%s/\n' "$SITE_URL"
    else
        printf '%s/%s\n' "$SITE_URL" "$filename"
    fi
}

page_metadata_field() {
    local filename="$1"
    local field="$2"
    local fallback="${3:-}"
    python3 - "$PAGE_METADATA_FILE" "$filename" "$field" "$fallback" <<'PY'
import json
import os
import sys

metadata_path, filename, field, fallback = sys.argv[1:5]
if not os.path.exists(metadata_path):
    print(fallback, end="")
    raise SystemExit(0)

with open(metadata_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

value = data.get("pages", {}).get(filename, {}).get(
    field,
    data.get("defaults", {}).get(field, fallback),
)

if isinstance(value, bool):
    print("true" if value else "false", end="")
elif value is None:
    print(fallback, end="")
else:
    print(str(value), end="")
PY
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
        page_title=$(sed -n 's/^<!--[[:space:]]*TITLE:[[:space:]]*\(.*\)[[:space:]]*-->$/\1/p' "$page" | head -n 1)
        nav_active=$(sed -n 's/^<!--[[:space:]]*NAV:[[:space:]]*\([[:alnum:]_-]*\)[[:space:]]*-->$/\1/p' "$page" | head -n 1)
        if [ -z "$page_title" ]; then
            page_title="Documentation"
        fi
        page_title=$(page_metadata_field "$filename" "title" "$page_title")
        page_description=$(page_metadata_field "$filename" "description" "C-Kernel-Engine documentation for a CPU-native AI runtime, kernels, code generation, training, and inference on Linux systems.")
        page_robots=$(page_metadata_field "$filename" "robots" "index,follow")

        # Read partials
        header=$(cat "$PARTIALS_DIR/header.html")
        footer=$(cat "$PARTIALS_DIR/footer.html")

        # Read page content (skip metadata comments)
        content=$(sed '/^<!--.*-->$/d' "$page")

        # Replace template variables in header
        header="${header//\{\{PAGE_TITLE\}\}/$page_title}"
        canonical_url=$(build_canonical_url "$filename")
        header="${header//\{\{CANONICAL_URL\}\}/$canonical_url}"
        header="${header//\{\{META_DESCRIPTION\}\}/$page_description}"
        header="${header//\{\{META_ROBOTS\}\}/$page_robots}"

        # Clear all nav active states
        header="${header//\{\{NAV_INDEX\}\}/}"
        header="${header//\{\{NAV_SPECTRUM\}\}/}"
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
        header="${header//\{\{NAV_THREADPOOL\}\}/}"

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

touch "$SCRIPT_DIR/.nojekyll"

echo "  Writing: robots.txt"
cat > "$SCRIPT_DIR/robots.txt" <<EOF
User-agent: *
Allow: /

Sitemap: $SITE_URL/sitemap.xml
EOF

echo "  Writing: sitemap.xml"
{
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    for built_page in "$SCRIPT_DIR"/*.html; do
        if [ -f "$built_page" ]; then
            built_name=$(basename "$built_page")
            include_in_sitemap=$(page_metadata_field "$built_name" "sitemap" "true")
            if [ "$include_in_sitemap" != "true" ]; then
                continue
            fi
            built_url=$(build_canonical_url "$built_name")
            echo '  <url>'
            echo "    <loc>$built_url</loc>"
            echo "    <lastmod>$CURRENT_DATE</lastmod>"
            echo '  </url>'
        fi
    done
    echo '</urlset>'
} > "$SCRIPT_DIR/sitemap.xml"

echo "Build complete! Generated files:"
ls -la "$SCRIPT_DIR"/*.html 2>/dev/null || echo "  No HTML files generated"

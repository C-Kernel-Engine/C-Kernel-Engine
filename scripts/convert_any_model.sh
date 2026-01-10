#!/bin/bash
# Simple script to convert ANY HF model to bump format
# Auto-detects architecture and converts

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage
usage() {
    echo -e "${BLUE}Usage:${NC} $0 <model_path> [output_path] [options]"
    echo ""
    echo "Convert any HuggingFace model to bump format"
    echo ""
    echo "Arguments:"
    echo "  model_path          Path to HF model directory"
    echo "  output_path        Output bump file (optional, defaults to model.bump)"
    echo ""
    echo "Options:"
    echo "  --arch ARCH        Architecture (llama, devstral, smollm, qwen, auto)"
    echo "  --dtype DTYPE      Data type (float32, q4_k, q4_k_m)"
    echo "  --context LEN      Override context length"
    echo "  --inspect          Only inspect model, don't convert"
    echo "  --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Devstral"
    echo "  $0 /path/to/LLaMA my_model.bump"
    echo "  $0 /path/to/Model --arch auto --dtype q4_k"
}

# Check arguments
if [ $# -lt 1 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    usage
    exit 0
fi

MODEL_PATH="$1"
OUTPUT_PATH="${2:-model.bump}"
shift || true
shift || true

# Parse options
ARCH="auto"
DTYPE="float32"
CONTEXT=""
INSPECT_ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --context)
            CONTEXT="--context $2"
            shift 2
            ;;
        --inspect)
            INSPECT_ONLY="1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check model path
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path does not exist: $MODEL_PATH${NC}"
    exit 1
fi

# Get model name
MODEL_NAME=$(basename "$MODEL_PATH")
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Model: $MODEL_NAME${NC}"
echo -e "${BLUE}Path: $MODEL_PATH${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Step 1: Inspect model
echo -e "${YELLOW}[1/2] Inspecting model...${NC}"
python scripts/inspect_model.py --checkpoint "$MODEL_PATH" 2>&1 | head -50

if [ -n "$INSPECT_ONLY" ]; then
    echo -e "\n${GREEN}Inspection complete. Use --inspect to disable this message.${NC}"
    exit 0
fi

# Step 2: Convert
echo -e "\n${YELLOW}[2/2] Converting to bump format...${NC}"

# Build command
CMD="python scripts/convert_to_bump_universal.py --checkpoint '$MODEL_PATH' --output '$OUTPUT_PATH' --arch $ARCH --dtype $DTYPE $CONTEXT"

echo -e "${BLUE}Running:${NC} $CMD"
echo ""

# Run conversion
if eval $CMD; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Conversion successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Output: $OUTPUT_PATH${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Test the model:"
    echo -e "     ck_cli_v5 --model $OUTPUT_PATH --prompt 'Hello'"
    echo ""
    echo -e "  2. Or use in your code:"
    echo -e "     # Load the bump file in your application"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Conversion failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  1. Check the error message above"
    echo -e "  2. Try with --arch auto"
    echo -e "  3. Or specify the architecture:"
    echo -e "     --arch llama     (for LLaMA)"
    echo -e "     --arch devstral  (for Devstral/SmolLM)"
    echo -e "     --arch qwen      (for Qwen)"
    echo ""
    echo -e "  4. For more help, see:"
    echo -e "     README-BUMP-CONVERTER.md"
    echo -e "     docs/bump-converter-multi-arch.md"
    exit 1
fi

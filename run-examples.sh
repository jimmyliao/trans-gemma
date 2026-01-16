#!/bin/bash
# Enhanced wrapper to run TranslateGemma examples with uv
# Usage:
#   ./run-examples.sh verify-hf-token    # Run verification
#   ./run-examples.sh local-test         # Run local test with cleanup option
#   ./run-examples.sh cleanup            # Clean up cache and temp files
#   ./run-examples.sh help               # Show help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show help
show_help() {
    echo "TranslateGemma Examples Runner"
    echo ""
    echo "Usage: ./run-examples.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  verify-hf-token       Verify Hugging Face token and model access (no GPU)"
    echo "  local-test            Full local test with optional cleanup (needs GPU)"
    echo "  translategemma-fix    TranslateGemma correct format example (needs GPU)"
    echo "  simple-translation    Cloud Run API client (no GPU)"
    echo "  cleanup               Clean up cache and temporary files"
    echo "  help                  Show this help message"
    echo ""
    echo "Device Options (for local-test, translategemma-fix):"
    echo "  --cpu                 Force CPU-only mode (slower, no GPU)"
    echo "  --mps, --metal        Force MPS/Metal mode (M1/M2/M3 Mac)"
    echo "  --auto                Auto-detect best device (default)"
    echo ""
    echo "Examples:"
    echo "  ./run-examples.sh verify-hf-token"
    echo "  ./run-examples.sh local-test"
    echo "  ./run-examples.sh local-test --cpu"
    echo "  ./run-examples.sh local-test --mps"
    echo "  ./run-examples.sh cleanup"
}

# Function to clean up caches
cleanup_caches() {
    echo -e "${BLUE}🧹 TranslateGemma Cache Cleanup${NC}"
    echo ""

    # Check Hugging Face cache
    if [ -d "$HOME/.cache/huggingface/hub" ]; then
        HF_SIZE=$(du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null | awk '{print $1}')
        echo -e "${YELLOW}📦 Hugging Face cache: $HF_SIZE${NC}"

        # Check for incomplete downloads
        INCOMPLETE_COUNT=$(find "$HOME/.cache/huggingface/hub" -name "*.incomplete" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$INCOMPLETE_COUNT" -gt 0 ]; then
            INCOMPLETE_SIZE=$(du -ch "$HOME/.cache/huggingface/hub"/**/blobs/*.incomplete 2>/dev/null | tail -1 | awk '{print $1}')
            echo -e "${YELLOW}   ⚠️  Found $INCOMPLETE_COUNT incomplete download(s): $INCOMPLETE_SIZE${NC}"
        fi
    fi

    # Check uv cache
    if [ -d "$HOME/.cache/uv" ]; then
        UV_SIZE=$(du -sh "$HOME/.cache/uv" 2>/dev/null | awk '{print $1}')
        echo -e "${YELLOW}📦 uv cache: $UV_SIZE${NC}"
    fi

    echo ""
    read -p "Do you want to clean up these caches? [y/N]: " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${GREEN}Cleaning up...${NC}"

        # Clean Hugging Face incomplete downloads
        if [ "$INCOMPLETE_COUNT" -gt 0 ]; then
            echo "🗑️  Removing incomplete downloads..."
            find "$HOME/.cache/huggingface/hub" -name "*.incomplete" -delete 2>/dev/null
            rm -rf "$HOME/.cache/huggingface/hub/.locks" 2>/dev/null
            echo -e "${GREEN}✅ Incomplete downloads removed${NC}"
        fi

        # Clean uv cache
        if command -v uv &> /dev/null; then
            echo "🗑️  Cleaning uv cache..."
            uv cache clean 2>&1 | grep "Removed\|freed" || echo "uv cache cleaned"
            echo -e "${GREEN}✅ uv cache cleaned${NC}"
        fi

        # Clean .venv if exists
        if [ -d ".venv" ]; then
            VENV_SIZE=$(du -sh .venv 2>/dev/null | awk '{print $1}')
            echo ""
            echo -e "${YELLOW}📦 Found virtual environment (.venv): $VENV_SIZE${NC}"
            read -p "Remove .venv? (can be recreated) [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf .venv
                echo -e "${GREEN}✅ .venv removed${NC}"
            fi
        fi

        echo ""
        echo -e "${GREEN}🎉 Cleanup complete!${NC}"

        # Show new cache sizes
        echo ""
        echo "📊 Current cache sizes:"
        [ -d "$HOME/.cache/huggingface/hub" ] && du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null | awk '{print "   Hugging Face: " $1}'
        [ -d "$HOME/.cache/uv" ] && du -sh "$HOME/.cache/uv" 2>/dev/null | awk '{print "   uv: " $1}'
    else
        echo "Cleanup cancelled."
    fi
}

# Function to parse device options
parse_device_option() {
    local device_option=""

    for arg in "$@"; do
        case "$arg" in
            --cpu)
                device_option="cpu"
                ;;
            --mps|--metal|--gpu)
                device_option="mps"
                ;;
            --auto)
                device_option="auto"
                ;;
        esac
    done

    echo "$device_option"
}

# Function to run example with post-execution cleanup option
run_with_cleanup_option() {
    local script_name=$1
    shift

    # Parse device option
    local device=$(parse_device_option "$@")

    # Set FORCE_DEVICE environment variable if specified
    if [ -n "$device" ]; then
        export FORCE_DEVICE="$device"
        echo -e "${BLUE}🎯 Device mode: $device${NC}"
        echo ""
    fi

    echo -e "${BLUE}Running $script_name...${NC}"
    echo ""

    # Run the script (remove device options from args)
    local clean_args=()
    for arg in "$@"; do
        case "$arg" in
            --cpu|--mps|--metal|--gpu|--auto)
                # Skip device options
                ;;
            *)
                clean_args+=("$arg")
                ;;
        esac
    done

    uv run python "examples/${script_name}.py" "${clean_args[@]}"

    # Ask for cleanup after execution
    echo ""
    echo -e "${YELLOW}Would you like to clean up caches now? (recommended after testing)${NC}"
    read -p "Clean up? [y/N]: " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        cleanup_caches
    fi
}

# Main script starts here

# Check for help
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Check for cleanup command
if [ "$1" = "cleanup" ]; then
    cleanup_caches
    exit 0
fi

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_TOKEN | xargs)
fi

# Check if HF_TOKEN is set (skip for cleanup)
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}❌ HF_TOKEN not found in .env or environment${NC}"
    echo ""
    echo "Please:"
    echo "  1. Copy .env.example to .env"
    echo "  2. Edit .env and add your HF_TOKEN"
    echo "  3. Run this script again"
    echo ""
    echo "Or run: ./run-examples.sh help"
    exit 1
fi

SCRIPT_NAME="${1:-verify-hf-token}"

# Check if script exists
if [ ! -f "examples/${SCRIPT_NAME}.py" ]; then
    echo -e "${RED}❌ Script not found: examples/${SCRIPT_NAME}.py${NC}"
    echo ""
    echo "Available scripts:"
    ls examples/*.py | sed 's/examples\//  /' | sed 's/.py$//'
    echo ""
    echo "Run: ./run-examples.sh help"
    exit 1
fi

# Run script based on type
case "$SCRIPT_NAME" in
    local-test|translategemma-fix)
        # Scripts that might benefit from cleanup afterward
        # Also support device options
        run_with_cleanup_option "$SCRIPT_NAME" "${@:2}"
        ;;
    *)
        # Other scripts run normally (but still support device options)
        local device=$(parse_device_option "${@:2}")
        if [ -n "$device" ]; then
            export FORCE_DEVICE="$device"
            echo -e "${BLUE}🎯 Device mode: $device${NC}"
            echo ""
        fi

        echo -e "${BLUE}Running $SCRIPT_NAME...${NC}"
        echo ""

        # Remove device options from args
        local clean_args=()
        for arg in "${@:2}"; do
            case "$arg" in
                --cpu|--mps|--metal|--gpu|--auto)
                    # Skip device options
                    ;;
                *)
                    clean_args+=("$arg")
                    ;;
            esac
        done

        uv run python "examples/${SCRIPT_NAME}.py" "${clean_args[@]}"

        # Show cleanup hint
        echo ""
        echo -e "${YELLOW}💡 Tip: Run './run-examples.sh cleanup' to free up cache space${NC}"
        ;;
esac

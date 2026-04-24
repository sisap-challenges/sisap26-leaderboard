#!/bin/bash

# SISAP25 Task 2 Data Setup Script (Bash version)
# Downloads required datasets for k-NN graph construction on GOOAQ dataset

set -e  # Exit on error

# Configuration
BASE_URL="https://huggingface.co/datasets/sadit/SISAP2025/resolve/main"
DATA_DIR="${1:-data}"
INCLUDE_EVAL="${2:-false}"

# File specifications
TASK2_FILES="allknn-benchmark-dev-gooaq.h5 benchmark-dev-gooaq.h5"
EVAL_FILE="benchmark-eval-gooaq.h5"

get_file_description() {
    case "$1" in
        "allknn-benchmark-dev-gooaq.h5")
            echo "768MB - Task 2 gold standard (k=15 nearest neighbors)"
            ;;
        "benchmark-dev-gooaq.h5")
            echo "4.8GB - Full development dataset with train/test splits"
            ;;
        "benchmark-eval-gooaq.h5")
            echo "7.9GB - Final evaluation dataset (optional)"
            ;;
        *)
            echo "Unknown file"
            ;;
    esac
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}SISAP25 Challenge - Task 2 Data Setup${NC}"
    echo -e "${BLUE}=====================================${NC}"
    echo
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

download_file() {
    local filename="$1"
    local description=$(get_file_description "$filename")
    local url="$BASE_URL/$filename"
    local filepath="$DATA_DIR/$filename"
    
    if [[ -f "$filepath" ]]; then
        print_success "$filename already exists, skipping download"
        return 0
    fi
    
    echo
    print_info "Downloading $filename"
    echo "    Description: $description"
    echo "    URL: $url"
    
    # Check if wget or curl is available
    if command -v wget &> /dev/null; then
        if wget --progress=bar:force -O "$filepath" "$url" 2>&1; then
            print_success "Successfully downloaded $filename"
            return 0
        else
            print_error "Failed to download $filename with wget"
            rm -f "$filepath"  # Clean up partial download
            return 1
        fi
    elif command -v curl &> /dev/null; then
        if curl -L --progress-bar -o "$filepath" "$url"; then
            print_success "Successfully downloaded $filename"
            return 0
        else
            print_error "Failed to download $filename with curl"
            rm -f "$filepath"  # Clean up partial download
            return 1
        fi
    else
        print_error "Neither wget nor curl is available for downloading"
        return 1
    fi
}

verify_file() {
    local filename="$1"
    local filepath="$DATA_DIR/$filename"
    
    if [[ ! -f "$filepath" ]]; then
        print_error "$filename: File not found"
        return 1
    fi
    
    # Check if it's a valid HDF5 file (basic check)
    if command -v file &> /dev/null; then
        if file "$filepath" | grep -q "HDF"; then
            local size=$(du -h "$filepath" | cut -f1)
            print_success "$filename: Valid HDF5 file ($size)"
            return 0
        else
            print_warning "$filename: May not be a valid HDF5 file"
            return 1
        fi
    else
        # Basic size check if file command is not available
        if [[ -s "$filepath" ]]; then
            local size=$(du -h "$filepath" | cut -f1)
            print_success "$filename: Downloaded ($size)"
            return 0
        else
            print_error "$filename: File is empty"
            return 1
        fi
    fi
}

show_usage() {
    echo "Usage: $0 [DATA_DIR] [INCLUDE_EVAL]"
    echo
    echo "Arguments:"
    echo "  DATA_DIR      Directory to store datasets (default: ./data)"
    echo "  INCLUDE_EVAL  Set to 'true' to download evaluation dataset (default: false)"
    echo
    echo "Examples:"
    echo "  $0                    # Download to ./data, skip eval dataset"
    echo "  $0 /path/to/data      # Download to custom directory"
    echo "  $0 data true          # Include evaluation dataset (7.9GB extra)"
    echo
    echo "Requirements:"
    echo "  - wget or curl for downloading"
    echo "  - ~6GB free disk space (12GB with evaluation dataset)"
    echo
}

show_dataset_info() {
    echo
    print_info "Task 2 Dataset Information"
    echo "========================="
    echo
    echo "🎯 Objective: k=15 nearest neighbor graph construction"
    echo "📊 Dataset: GOOAQ (Google Questions & Answers)"
    echo "📏 Dimensions: 384 (sentence-BERT embeddings)"
    echo "📦 Vectors: 3,012,496 (3M)"
    echo "📐 Similarity: Cosine similarity / dot product"
    echo "🎯 Target recall: ≥ 0.8"
    echo "💻 Resources: 8 CPUs, 16GB RAM, 12h limit"
    echo
    echo "📁 File Usage:"
    echo "  • allknn-benchmark-dev-gooaq.h5: Gold standard for evaluation"
    echo "  • benchmark-dev-gooaq.h5: Main dataset for development" 
    echo "  • benchmark-eval-gooaq.h5: Final submission evaluation"
    echo
}

main() {
    # Check arguments
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    print_header
    
    # Create data directory
    mkdir -p "$DATA_DIR"
    print_info "Data directory: $(realpath "$DATA_DIR")"
    
    # Download required files
    local success=true
    
    # Always download required files
    for file in $TASK2_FILES; do
        if ! download_file "$file"; then
            success=false
        fi
    done
    
    # Optionally download evaluation dataset
    if [[ "$INCLUDE_EVAL" == "true" ]]; then
        if ! download_file "$EVAL_FILE"; then
            success=false
        fi
    else
        print_info "Skipping evaluation dataset (use '$0 $DATA_DIR true' to include)"
    fi
    
    echo
    print_info "Verifying downloaded files..."
    
    # Verify files
    local verify_success=true
    for file in $TASK2_FILES; do
        filepath="$DATA_DIR/$file"
        if [[ -f "$filepath" ]]; then
            if ! verify_file "$file"; then
                verify_success=false
            fi
        else
            print_error "$file: Missing"
            verify_success=false
        fi
    done
    
    # Check eval file if it should exist
    if [[ "$INCLUDE_EVAL" == "true" ]]; then
        filepath="$DATA_DIR/$EVAL_FILE"
        if [[ -f "$filepath" ]]; then
            if ! verify_file "$EVAL_FILE"; then
                verify_success=false
            fi
        else
            print_error "$EVAL_FILE: Missing"
            verify_success=false
        fi
    fi
    
    echo
    if $success && $verify_success; then
        print_success "Data setup completed successfully!"
        show_dataset_info
        exit 0
    else
        print_error "Data setup failed or verification errors occurred"
        exit 1
    fi
}

# Run main function
main "$@"
#!/bin/bash

# GPU Testing Script for Plonky2
# This script validates CUDA setup, zeknox library, and runs GPU-accelerated tests

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Plonky2 GPU Testing Script"
echo "========================================="
echo ""

# Step 1: Check NVIDIA driver and CUDA
echo -e "${YELLOW}[1/7] Checking NVIDIA driver and CUDA...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

echo "NVIDIA Driver Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Check for CUDA toolkit
if command -v nvcc &> /dev/null; then
    echo "CUDA Compiler Version:"
    nvcc --version | grep "release"
    echo ""
elif [ -n "$CUDA_HOME" ]; then
    echo "CUDA_HOME is set to: $CUDA_HOME"
    echo ""
else
    echo -e "${YELLOW}WARNING: nvcc not found and CUDA_HOME not set. CUDA toolkit may not be installed.${NC}"
    echo "Continuing anyway as runtime libraries may still be available..."
    echo ""
fi

echo -e "${GREEN} NVIDIA driver check passed${NC}"
echo ""

# Step 2: Check zeknox library
echo -e "${YELLOW}[2/7] Checking zeknox library...${NC}"
ZEKNOX_PATH="../zeknox"
if [ ! -d "$ZEKNOX_PATH" ]; then
    echo -e "${RED}ERROR: zeknox library not found at $ZEKNOX_PATH${NC}"
    echo "Expected location: $(cd .. && pwd)/zeknox"
    exit 1
fi

if [ -d "$ZEKNOX_PATH/wrappers/rust" ]; then
    echo "Found zeknox library at: $(cd $ZEKNOX_PATH && pwd)"
    echo "Rust wrappers directory exists: $ZEKNOX_PATH/wrappers/rust"
else
    echo -e "${YELLOW}WARNING: zeknox/wrappers/rust not found, but zeknox directory exists${NC}"
fi

echo -e "${GREEN} zeknox library check passed${NC}"
echo ""

# Step 3: Run field tests
echo -e "${YELLOW}[3/7] Running field tests with GPU acceleration...${NC}"
echo "Command: cd field && cargo test --release --features=cuda -- --test-threads=1"
echo ""

cd field
if cargo test --release --features=cuda -- --test-threads=1; then
    echo ""
    echo -e "${GREEN} Field tests passed${NC}"
else
    echo ""
    echo -e "${RED}ERROR: Field tests failed${NC}"
    cd ..
    exit 1
fi
cd ..
echo ""

# Step 4: Run fibonacci example with CUDA for correctness
echo -e "${YELLOW}[4/7] Running fibonacci example with CUDA features...${NC}"
echo "Command: NUM_OF_GPUS=1 cargo run --release --features=cuda_sanity_check --example fibonacci"
echo ""

if NUM_OF_GPUS=1 cargo run --release --features=cuda_sanity_check --example fibonacci; then
    echo ""
    echo -e "${GREEN} Fibonacci example completed successfully with GPU${NC}"
else
    echo ""
    echo -e "${RED}ERROR: Fibonacci example failed with GPU${NC}"
    exit 1
fi
echo ""

# Step 5: Run fibonacci example with CUDA for speed
echo -e "${YELLOW}[5/7] Running fibonacci example with CUDA features...${NC}"
echo "Command: NUM_OF_GPUS=1 cargo run --release --example fibonacci --features=cuda"
echo ""

if NUM_OF_GPUS=1 cargo run --release --example fibonacci --features=cuda; then
    echo ""
    echo -e "${GREEN} Fibonacci example completed successfully with GPU${NC}"
else
    echo ""
    echo -e "${RED}ERROR: Fibonacci example failed with GPU${NC}"
    exit 1
fi
echo ""


# Step 6: Run fibonacci example with CPU
echo -e "${YELLOW}[6/7] Running fibonacci example with CUDA features...${NC}"
echo "Command: NUM_OF_GPUS=1 cargo run --release --example fibonacci"
echo ""

if cargo run --release --example fibonacci; then
    echo ""
    echo -e "${GREEN} Fibonacci example completed successfully with CPU${NC}"
else
    echo ""
    echo -e "${RED}ERROR: Fibonacci example failed with CPU${NC}"
    exit 1
fi
echo ""

# Step 7: Summary
echo "========================================="
echo -e "${GREEN}All GPU tests completed successfully!${NC}"
echo "========================================="
echo ""
echo "Tests run:"
echo "   NVIDIA driver and CUDA verification"
echo "   zeknox library verification"
echo "   Field tests (FFT, polynomials, interpolation, cosets)"
echo "   Fibonacci proof generation with GPU acceleration"
echo ""
echo -e "${GREEN}GPU testing complete!${NC}"

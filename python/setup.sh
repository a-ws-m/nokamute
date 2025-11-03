#!/bin/bash
# Setup script for Nokamute Python bindings

set -e

echo "========================================="
echo "Nokamute Python Bindings Setup"
echo "========================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed"
    echo "Please install from https://rustup.rs"
    exit 1
fi

echo "✓ Rust found: $(rustc --version)"

# Navigate to python directory
cd "$(dirname "$0")"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Build and install Rust bindings
echo ""
echo "Building Rust bindings with maturin..."
maturin develop --release

# Test the installation
echo ""
echo "Testing installation..."
python -c "import nokamute; print('✓ nokamute module imported successfully')"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run the example:"
echo "  python example.py"
echo ""
echo "To start training:"
echo "  python train.py --games 100 --iterations 10"
echo ""
echo "To evaluate a model:"
echo "  python evaluate.py --model checkpoints/model_latest.pt --mode vs-random"
echo ""

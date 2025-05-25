#!/bin/tcsh

# Setup script for GenomicLightning framework
# This script will create a virtual environment, install dependencies,
# and provide instructions for getting started.

echo "\n===== Setting up GenomicLightning environment =====\n"

# Check if Python is available
which python3 >& /dev/null
if ($status != 0) then
    echo "ERROR: Python 3 is required but not found. Please install Python 3.8 or newer."
    exit 1
endif

# Get Python version
set PYTHON_VERSION = `python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2`
set REQUIRED_VERSION = 3.8

# Compare versions (simple string comparison works for this format)
if ("$PYTHON_VERSION" < "$REQUIRED_VERSION") then
    echo "ERROR: Python $REQUIRED_VERSION+ is required, but found Python $PYTHON_VERSION"
    exit 1
endif

# Create a virtual environment if it doesn't exist
if (! -d "venv") then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if ($status != 0) then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    endif
endif

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate.csh

# Install dependencies
echo "Installing GenomicLightning and dependencies..."
pip install -e ".[dev,logging]"
if ($status != 0) then
    echo "ERROR: Failed to install dependencies."
    exit 1
endif

# Check for PyTorch with CUDA
echo "Checking PyTorch installation..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Create a test dataset directory
if (! -d "data") then
    mkdir -p data/synthetic
endif

# Generate synthetic dataset for testing
echo "\nGenerating synthetic dataset for testing..."
python3 -c "
import os
import h5py
import numpy as np

print('Creating synthetic genomic dataset...')
file_path = 'data/synthetic/test_data.h5'

# Create sample size and dimensions
n_samples = 1000
seq_length = 1000
n_targets = 919

# Create random one-hot encoded sequences
sequences = np.zeros((n_samples, 4, seq_length), dtype=np.float32)
for i in range(n_samples):
    for j in range(seq_length):
        nuc = np.random.randint(0, 4)
        sequences[i, nuc, j] = 1.0

# Create random targets (binary)
targets = np.random.randint(0, 2, size=(n_samples, n_targets)).astype(np.float32)

# Save to HDF5 file
with h5py.File(file_path, 'w') as f:
    f.create_dataset('sequences', data=sequences)
    f.create_dataset('targets', data=targets)
    
print(f'Saved synthetic dataset to {file_path}')
print(f'  - Samples: {n_samples}')
print(f'  - Sequence length: {seq_length}')
print(f'  - Targets: {n_targets}')
"

# Success message and instructions
echo "\n===== GenomicLightning setup complete! =====\n"
echo "To activate the virtual environment:"
echo "  source venv/bin/activate.csh"
echo ""
echo "Try running an example:"
echo "  python examples/simple_training_example.py"
echo ""
echo "Or use the synthetic dataset we just created:"
echo "  python examples/large_data_training_example.py --train_shards data/synthetic/test_data.h5 --val_shards data/synthetic/test_data.h5 --model_type danq --max_epochs 5"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "For more information, see README.md"

#!/bin/tcsh

echo "Setting up environment for GenomicLightning..."

# Load any required modules if on cluster
if (-f /etc/profile.d/modules.csh) then
    source /etc/profile.d/modules.csh
    module load python/3.9
    module load cuda
    echo "Loaded Python and CUDA modules"
endif

# Create virtual environment if it doesn't exist
if (! -d ~/genomic_env) then
    echo "Creating virtual environment..."
    python -m venv ~/genomic_env
endif

# Activate virtual environment
echo "Activating virtual environment..."
source ~/genomic_env/bin/activate.csh

# Install GenomicLightning
echo "Installing GenomicLightning..."
cd ~/GenomicLightning
pip install -e .

# Install additional dependencies
echo "Installing additional dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning
pip install h5py scikit-learn matplotlib seaborn pandas pyyaml

# Make sure UAVarPrior is available (if needed)
if (-d ~/UAVarPrior) then
    echo "Adding UAVarPrior to environment..."
    cd ~/UAVarPrior
    pip install -e .
endif

# Make sure FuGEP is available (if needed)
if (-d ~/FuGEP) then
    echo "Adding FuGEP to environment..."
    cd ~/FuGEP
    pip install -e .
endif

echo "Environment setup complete!"
echo "To activate the environment, use: source ~/genomic_env/bin/activate.csh"

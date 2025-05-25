#!/bin/tcsh
#SBATCH --job-name=genomic_train
#SBATCH --output=genomic_train_%j.out
#SBATCH --error=genomic_train_%j.err
#SBATCH --partition=high-gpu-small
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Print some information about the job
echo "Running on host: `hostname`"
echo "Time: `date`"
echo "Directory: `pwd`"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/GenomicLightning/setup_genomic_env.csh

# Create data directory if it doesn't exist
if (! -d ~/genomic_data) then
    mkdir -p ~/genomic_data
endif

# Generate sample data if it doesn't exist
if (! -f ~/genomic_data/train_shard1.h5) then
    echo "Generating sample data..."
    python ~/GenomicLightning/scripts/generate_sample_data.py \
        --output-dir ~/genomic_data \
        --num-train 10000 \
        --num-val 2000 \
        --num-test 2000 \
        --seq-length 1000 \
        --num-targets 919
endif

# Create output directory if it doesn't exist
if (! -d ~/genomic_results) then
    mkdir -p ~/genomic_results
endif

# Run training script
echo "Starting model training..."
python ~/GenomicLightning/scripts/train_genomic_model.py \
    --model-type deepsea \
    --num-targets 919 \
    --data-dir ~/genomic_data \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --max-epochs 10 \
    --output-dir ~/genomic_results

echo "Job completed at: `date`"

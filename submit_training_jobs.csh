#!/bin/tcsh

# Make scripts executable
chmod +x ~/GenomicLightning/setup_genomic_env.csh
chmod +x ~/GenomicLightning/scripts/generate_sample_data.py
chmod +x ~/GenomicLightning/scripts/train_genomic_model.py
chmod +x ~/GenomicLightning/scripts/train_genomic_job.sbatch
chmod +x ~/GenomicLightning/scripts/train_danq_job.sbatch

# Setup environment first (this also installs the package)
echo "Setting up environment..."
source ~/GenomicLightning/setup_genomic_env.csh

# Submit jobs
echo "Submitting DeepSEA training job..."
sbatch ~/GenomicLightning/scripts/train_genomic_job.sbatch

echo "Submitting DanQ training job..."
sbatch ~/GenomicLightning/scripts/train_danq_job.sbatch

echo "Jobs submitted. You can check their status with: 'squeue -u $USER'"

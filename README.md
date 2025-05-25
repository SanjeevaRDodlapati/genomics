# GenomicLightning

A PyTorch Lightning framework for genomic deep learning models, designed with modern software architecture principles.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0%2B-792ee5)](https://www.pytorchlightning.ai/)

## Features

- **Modern Architecture**: Modular, composable components built on PyTorch Lightning
- **Multiple Models**: Support for DeepSEA, DanQ, ChromDragoNN, and custom architectures
- **Efficient Data Handling**: Streaming and sharding for large genomic datasets
- **Interpretability**: Advanced visualization tools for model interpretation
- **Specialized Metrics**: Metrics designed for genomic applications
- **Variant Analysis**: Tools for predicting and analyzing variant effects
- **Legacy Integration**: Seamless import from UAVarPrior/FuGEP models

## Installation

```bash
# Development installation
pip install -e ".[dev]"

# With logging tools
pip install -e ".[logging]"

# With all optional dependencies
pip install -e ".[dev,logging]"
```

## Quick Start

```bash
# Train a model
genomic_lightning train configs/example_deepsea.yml

# Evaluate a model
genomic_lightning evaluate configs/example_deepsea.yml --ckpt path/to/checkpoint.ckpt

# Train on large sharded data
python examples/large_data_training_example.py --train_shards data/train/*.h5 --val_shards data/val/*.h5 --model_type danq

# Run with model interpretability
python examples/large_data_training_example.py --train_shards data/train/*.h5 --val_shards data/val/*.h5 --interpret
```

## Supported Models

GenomicLightning includes implementations of several state-of-the-art genomic deep learning models:

- **DeepSEA** - Convolutional neural network for predicting chromatin effects of sequence alterations
- **DanQ** - Hybrid CNN-RNN architecture that captures both local motifs and dependencies
- **ChromDragoNN** - Residual network architecture for predicting chromatin features from DNA sequence

## Large Data Support

For working with large genomic datasets, use the sharded data module:

```python
from genomic_lightning.data.sharded_data_module import ShardedGenomicDataModule

data_module = ShardedGenomicDataModule(
    train_shards=["data/train/shard1.h5", "data/train/shard2.h5"],
    val_shards=["data/val/shard1.h5"],
    batch_size=32,
    cache_size=1000  # Number of samples to cache in memory
)
```

## Interpretability Tools

Visualize and interpret what your models have learned:

```python
from genomic_lightning.visualization.motif_visualization import MotifVisualizer

# Create visualizer
visualizer = MotifVisualizer(model)

# Extract and visualize motifs from convolutional filters
visualizer.save_filter_logos(model, output_dir="motifs")

# Generate attribution maps using integrated gradients
attributions = visualizer.get_integrated_gradients(sequences, target_class=0)
fig = visualizer.visualize_sequence_attribution(sequences[0], attributions[0])
```

## Project Structure

- `genomic_lightning/models/` - Neural network architectures
- `genomic_lightning/data/` - Data loading and processing
- `genomic_lightning/lightning_modules/` - PyTorch Lightning modules
- `genomic_lightning/callbacks/` - Custom Lightning callbacks
- `genomic_lightning/cli/` - Command line interface
- `genomic_lightning/config/` - Configuration utilities
- `genomic_lightning/utils/` - Shared utilities
- `genomic_lightning/metrics/` - Specialized metrics for genomic data
- `genomic_lightning/visualization/` - Tools for model interpretability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

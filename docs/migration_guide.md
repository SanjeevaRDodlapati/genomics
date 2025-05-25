# PyTorch Lightning Migration Guide

This guide provides instructions for migrating models and code from UAVarPrior and FuGEP to the new GenomicLightning framework.

## Why Migrate?

GenomicLightning offers several advantages over the legacy codebases:

1. **Modern Architecture**: Clean separation of concerns using PyTorch Lightning
2. **Better Performance**: Optimized data loading and processing
3. **Ease of Use**: Simplified APIs and configuration
4. **Enhanced Features**: Built-in visualization and interpretability tools
5. **Maintainability**: Modern codebase following best practices

## Migration Path

### Step 1: Identify Components to Migrate

Start by identifying which components you want to migrate:

- **Models**: Neural network architectures
- **Data Loading**: Samplers and datasets
- **Training Logic**: Training loops and optimization
- **Inference**: Prediction and evaluation

### Step 2: Import Legacy Models

GenomicLightning provides utilities to import models from UAVarPrior/FuGEP:

```python
from genomic_lightning.utils.legacy_import import import_fugep_model, import_uavarprior_model

# Import from UAVarPrior
legacy_model = import_uavarprior_model(
    model_path="/path/to/uavarprior/model.pth",
    model_type="deepsea"  # or "custom" with class_definition provided
)

# Import from FuGEP
legacy_model = import_fugep_model(
    model_path="/path/to/fugep/model.pth",
    model_config="/path/to/fugep/config.yml"
)
```

### Step 3: Wrap Legacy Models

Convert imported models to Lightning modules:

```python
from genomic_lightning.utils.wrapper_conversion import wrap_model_with_lightning

# Create a Lightning module from a legacy model
lightning_module = wrap_model_with_lightning(
    legacy_model,
    learning_rate=1e-4,
    loss_function="binary_cross_entropy",
    metrics=["auroc", "auprc"]
)
```

### Step 4: Adapt Legacy Samplers

Adapt legacy data samplers to work with GenomicLightning:

```python
from genomic_lightning.data.sampler_adapter import LegacySamplerAdapter
from genomic_lightning.data.sampler_data_module import SamplerDataModule

# Create a dataset adapter for a legacy sampler
dataset = LegacySamplerAdapter(
    legacy_sampler=your_legacy_sampler,
    batch_size=32
)

# Create a Lightning data module
data_module = SamplerDataModule(
    train_dataset=dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=32,
    num_workers=4
)
```

### Step 5: Train Using Lightning

Train your model using PyTorch Lightning:

```python
import pytorch_lightning as pl

# Create a trainer
trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    precision=16  # Mixed precision for faster training
)

# Train the model
trainer.fit(lightning_module, data_module)
```

## Example: Full Migration Workflow

Here's a complete example of migrating a UAVarPrior model:

```python
import pytorch_lightning as pl
from genomic_lightning.utils.legacy_import import import_uavarprior_model
from genomic_lightning.utils.wrapper_conversion import wrap_model_with_lightning
from genomic_lightning.data.sampler_adapter import LegacySamplerAdapter
from genomic_lightning.data.sampler_data_module import SamplerDataModule

# Import a legacy model
legacy_model = import_uavarprior_model(
    model_path="/path/to/uavarprior/model.pth",
    model_type="deepsea"
)

# Wrap with Lightning module
lightning_module = wrap_model_with_lightning(
    legacy_model,
    learning_rate=1e-4,
    loss_function="binary_cross_entropy",
    metrics=["auroc"]
)

# Create datasets from legacy samplers
train_dataset = LegacySamplerAdapter(your_legacy_train_sampler, batch_size=32)
val_dataset = LegacySamplerAdapter(your_legacy_val_sampler, batch_size=32)

# Create data module
data_module = SamplerDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=32,
    num_workers=4
)

# Create trainer
trainer = pl.Trainer(max_epochs=100, gpus=1)

# Train the model
trainer.fit(lightning_module, data_module)

# Save the model
trainer.save_checkpoint("migrated_model.ckpt")
```

## Migrating Custom Models

If you have custom model architectures:

1. Create a new model class in GenomicLightning
2. Copy your model architecture, updating any deprecated PyTorch functionality
3. Create a Lightning module for your model
4. Test to ensure equivalence with the original model

## Converting Predictions

GenomicLightning uses a different format for storing predictions. Convert legacy prediction formats:

```python
from genomic_lightning.utils.prediction_utils import convert_legacy_predictions

# Convert predictions from legacy format
lightning_predictions = convert_legacy_predictions(
    legacy_predictions,
    from_format="uavarprior",  # or "fugep"
    to_format="hdf5"           # or "numpy", "csv"
)
```

## Need Help?

If you encounter issues during migration, please:

1. Check the documentation in the codebase
2. Look at example scripts in the `examples/` directory
3. Refer to integration examples in `examples/uavarprior_integration_example.py`

## Full Framework Adoption

For new projects, consider using GenomicLightning's native models and data modules instead of migrating legacy code:

```python
from genomic_lightning.models.danq import DanQModel
from genomic_lightning.lightning_modules.danq import DanQLightningModule
from genomic_lightning.data.data_modules import GenomicDataModule

# Create model directly
model = DanQModel(...)
lightning_module = DanQLightningModule(model=model, ...)

# Use native data modules
data_module = GenomicDataModule(...)

# Train as usual
trainer = pl.Trainer(...)
trainer.fit(lightning_module, data_module)
```

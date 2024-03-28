# torch-utils
[![Downloads](https://static.pepy.tech/badge/pytorch-utilities)](https://pepy.tech/project/pytorch-utilities)

This repository contains useful functions and classes for Deep Learning engineers using PyTorch.

# Installation

You can install this package using pip. The name of the package in PyPI is **pytorch-utilities**:

`pip install pytorch-utilities`

## Cosine Annealing with Linear Warmup Learning Rate

Using this scheduler is as simple as using a default PyTorch scheduler.

Example usage:

```python
import torch
from torch.optim import AdamW
from torchutils.schedulers import CosineAnnealingLinearWarmup


# Initialize your model and dataloader
# model = ...
# dataloader = ...
# loss_fn = ...

# Initialize the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=0.0005)
scheduler = CosineAnnealingLinearWarmup(optimizer, warmup_epochs=5, max_epochs=100)

# If you want to step the scheduler after each iteration (batch), adjust the warmup_epochs and max_epochs accordingly
# scheduler = CosineAnnealingLinearWarmup(optimizer, warmup_epochs=5 * len(dataloader), max_epochs=100 * len(dataloader))

# Training loop
for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
  
        # Forward pass
        outputs = model(inputs)
  
        # Compute loss
        loss = loss_fn(outputs, targets)
  
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # If you want to step the scheduler after each iteration (batch), uncomment the following line
        # scheduler.step()
  
    # If you're stepping the scheduler after each epoch, do it here
    scheduler.step()
```

## Layer-wise Learning Rate Decay

Using `layerwise_lrd`, you can set different learning rates for different layers in your model, from the first layer to the last in an ascending order. This is a widely used fine-tuning technique in Deep Vision models that ensures the model keeps most of its learned parameters in the first layers, as the features extracted in these layers are usually low level such as edges and shapes which are beneficial in most image domains and do not need much of a change.

Currently, only ViT models implemented in `timm`, or with layer names like the ones implemented in it are supported.

Example Usage:

```python
import timm
import torch
from torchutils.schedulers import layerwise_lrd


# Load the model
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', num_classes=1000)

# Fetch model's parameter groups (in place of `model.parameters()`)
param_groups = layerwise_lrd(
    model,
    weight_decay=0.05,
    no_weight_decay_list=model.no_weight_decay(),
    layer_decay=0.75,
)

# Set the optimizer
optimizer = torch.optim.AdamW(param_groups, lr=0.001)

# Rest of your training code as usual
# ...
```

# torch-utils
This repository contains useful functions and classes for Deep Learning engineers using PyTorch.

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

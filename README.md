# pytorch-parametrizations

## Spectral Parametrization

    This module provides a PyTorch implementation of the spectral parametrization of the weights of a 2D convolutional layer, as introduced in the paper "Efficient Nonlinear Transforms for Lossy Image Compression" by Johannes Ballé, PCS 2018.

### Usage

```python
import torch
import torch.nn.utils.parametrize as parametrize

from pytorch_parametrizations import SpectralParametrization

# Create a 2D convolutional layer
conv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

# Register the spectral parametrization for the weights of the layer
parametrize.register_parametrization(conv, 'weight', SpectralParametrization(conv.kernel_size), unsafe=True)


print(conv.parametrizations.weight)
# Output:
# conv.parametrizations.weight
# ParametrizationList(
#   (0): SpectralParametrization()
# )
```

## register_spectral_parametrization

    This function registers the spectral parametrization for every Conv2d and ConvTranspose2d layer in a given module.

### Usage

```python
import torch
from pytorch_parametrizations.spectral.utils import register_spectral_parametrization

# Create a module
module = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
    torch.nn.Sigmoid()
)

# Register the spectral parametrization for every Conv2d and ConvTranspose2d layer in the module
register_spectral_parametrization(module)

print(module)
# Output:
# Sequential(
#   (0): ParametrizedConv2d(
#     3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
#     (parametrizations): ModuleDict(
#       (weight): ParametrizationList(
#         (0): SpectralParametrization()
#       )
#     )
#   )
#   (1): ReLU()
#   (2): ParametrizedConv2d(
#     64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
#     (parametrizations): ModuleDict(
#       (weight): ParametrizationList(
#         (0): SpectralParametrization()
#       )
#     )
#   )
#   (3): ReLU()
#   (4): ParametrizedConv2d(
#     64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
#     (parametrizations): ModuleDict(
#       (weight): ParametrizationList(
#         (0): SpectralParametrization()
#       )
#     )
#   )
#   (5): Sigmoid()
# )

# Unregister the spectral parametrization for every Conv2d and ConvTranspose2d layer in the module
register_spectral_parametrization(module, undo=True)

print(module)
# Output:
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU()
#   (4): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (5): Sigmoid()
# )
```

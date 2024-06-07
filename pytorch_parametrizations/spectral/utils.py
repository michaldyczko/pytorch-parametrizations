import torch
import torch.nn.utils.parametrize as parametrize

from pytorch_parametrizations.spectral.spectral import SpectralParametrization

__all__ = ["register_spectral_parametrization"]


def register_spectral_parametrization(
    model: torch.nn.Module, undo: bool = False
) -> None:
    """
    Register or remove spectral parametrizations for convolutional layers in the given model.

    Args:
        model (torch.nn.Module): The model to register or remove spectral parametrizations for.
        undo (bool, optional): If True, removes the spectral parametrizations. If False (default), registers the spectral parametrizations.

    Returns:
        None
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            if not undo:
                parametrize.register_parametrization(
                    module,
                    "weight",
                    SpectralParametrization(module.kernel_size),
                    unsafe=True,
                )
            else:
                parametrize.remove_parametrizations(module, "weight")

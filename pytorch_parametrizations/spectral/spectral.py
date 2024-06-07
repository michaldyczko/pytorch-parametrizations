import torch
from torch import nn


class SpectralParametrization(nn.Module):
    """
    SpectralParametrization is a PyTorch module that performs spectral parametrization
    using the Fast Fourier Transform (FFT) operations.

    Args:
        kernel_size (tuple): The size of the kernel used for FFT operations.

    Attributes:
        dim (tuple): The dimensions along which FFT operations are performed.
        kernel_size (tuple): The size of the kernel used for FFT operations.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.dim = (-2, -1)
        self.kernel_size = kernel_size

    def forward(self, X):
        """
        Performs forward pass of the spectral parametrization.

        Args:
            X (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after performing FFT operations.
        """
        R, I = torch.unbind(X)
        C = torch.complex(R, I)
        Y = torch.fft.irfftn(C, s=self.kernel_size, dim=self.dim, norm="ortho")
        return Y

    def right_inverse(self, Y):
        """
        Performs right inverse of the spectral parametrization.

        Args:
            Y (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after performing inverse FFT operations.
        """
        C = torch.fft.rfftn(Y, s=self.kernel_size, dim=self.dim, norm="ortho")
        R, I = C.real, C.imag
        X = torch.stack((R, I))
        return X

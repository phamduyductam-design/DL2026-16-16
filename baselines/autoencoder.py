"""Compact convolutional autoencoder baseline."""

import torch
from torch import nn


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, channels: int = 3, latent_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


def reconstruction_map(image: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(image - reconstruction), dim=1)


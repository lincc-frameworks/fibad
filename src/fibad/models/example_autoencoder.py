# ruff: noqa: D101, D102

# This example model is taken from the autoenocoder tutorial here
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.optim as optim
import torch.utils.data.dataloader
from torchvision.transforms.v2 import CenterCrop

# extra long import here to address a circular import issue
from fibad.models.model_registry import fibad_model


@fibad_model
class ExampleAutoencoder(nn.Module):
    def __init__(self, model_config, shape=(5, 250, 250)):
        super().__init__()
        self.config = model_config

        # TODO xcxc config-ize or get from data loader somehow
        self.num_input_channels, self.image_width, self.image_height = shape

        self.c_hid = self.config.get("base_channel_size", 32)
        self.latent_dim = self.config.get("latent_dim", 64)

        # Calculate how much our convolutional layers will affect the size of final convolution
        # Formula evaluated from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #
        # If the number of layers are changed this will need to be rewritten.
        self.conv_end_w = self.conv2d_multi_layer(self.image_width, 3, kernel_size=3, padding=1, stride=2)
        self.conv_end_h = self.conv2d_multi_layer(self.image_height, 3, kernel_size=3, padding=1, stride=2)

        self._init_encoder()
        self._init_decoder()

    def conv2d_multi_layer(self, input_size, num_applications, **kwargs) -> int:
        for _ in range(num_applications):
            input_size = self.conv2d_output_size(input_size, **kwargs)

        return input_size

    def conv2d_output_size(self, input_size, kernel_size, padding=0, stride=1, dilation=1) -> int:
        # From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        return int((numerator / stride) + 1)

    def _init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_input_channels, self.c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * self.conv_end_h * self.conv_end_w * self.c_hid, self.latent_dim),
        )

    def _eval_encoder(self, x):
        return self.encoder(x)

    def _init_decoder(self):
        self.dec_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * self.conv_end_h * self.conv_end_w * self.c_hid), nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                2 * self.c_hid, 2 * self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                2 * self.c_hid, self.c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.c_hid, self.num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, so the output has to be bounded as well
        )

    def _eval_decoder(self, x):
        x = self.dec_linear(x)
        x = x.reshape(x.shape[0], -1, self.conv_end_h, self.conv_end_w)
        x = self.decoder(x)
        x = CenterCrop(size=(self.image_width, self.image_height))(x)
        return x

    def forward(self, x):
        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)
        return x_hat

    def train(self, trainloader, device=None):
        self.optimizer = self._optimizer()

        torch.set_grad_enabled(True)

        # print(f"len(trainloder) = {len(trainloader)}")
        for epoch in range(self.config.get("epochs", 2)):
            running_loss = 0.0
            for batch_num, data in enumerate(trainloader, 0):
                # When we run on a supervised dataset like CIFAR10, drop the labels given by the data loader
                x = data[0] if isinstance(data, tuple) else data

                x = x.to(device)
                x_hat = self.forward(x)
                loss = F.mse_loss(x, x_hat, reduction="none")
                loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                running_loss += loss.item()

                # Log every 2000 batches in an epoch, or the end of the epoch
                # Ensure we get one log message at the end of every epoch even if the
                # data size is less than 2000.
                log_freq = np.min([2000, len(trainloader)])
                if batch_num % log_freq == log_freq - 1:
                    print(f"[{epoch + 1}, {batch_num + 1}] loss: {running_loss / 2000}")
                    running_loss = 0.0

    def _optimizer(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def save(self):
        torch.save(self.state_dict(), self.config.get("weights_filepath", "example_autoencoder.pth"))

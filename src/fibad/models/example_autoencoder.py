# ruff: noqa: D101, D102

# This example model is taken from the autoenocoder tutorial here
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html

# The train function has been converted into train_step for use with pytorch-ignite.

import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.optim as optim
from torchvision.transforms.v2 import CenterCrop

# extra long import here to address a circular import issue
from fibad.models.model_registry import fibad_model


@fibad_model
class ExampleAutoencoder(nn.Module):
    def __init__(self, config, shape=(5, 250, 250)):
        super().__init__()
        self.config = config

        # TODO xcxc config-ize or get from data loader somehow
        self.num_input_channels, self.image_width, self.image_height = shape

        self.c_hid = self.config["model"]["base_channel_size"]
        self.latent_dim = self.config["model"]["latent_dim"]

        # Calculate how much our convolutional layers will affect the size of final convolution
        # Formula evaluated from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #
        # If the number of layers are changed this will need to be rewritten.
        self.conv_end_w = self.conv2d_multi_layer(self.image_width, 3, kernel_size=3, padding=1, stride=2)
        self.conv_end_h = self.conv2d_multi_layer(self.image_height, 3, kernel_size=3, padding=1, stride=2)

        self._init_encoder()
        self._init_decoder()

        # create this here for use in `train_step`, to avoid recreating at each step.
        self.optimizer = self._optimizer()

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

    def forward(self, batch):
        # When we run on a supervised dataset like CIFAR10, drop the labels given by the data loader
        x = batch[0] if isinstance(batch, tuple) else batch
        return self._eval_encoder(x)

    def train_step(self, batch):
        """This function contains the logic for a single training step. i.e. the
        contents of the inner loop of a ML training process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the inputs and labels for the current batch.

        Returns
        -------
        Current loss value
            The loss value for the current batch.
        """
        # When we run on a supervised dataset like CIFAR10, drop the labels given by the data loader
        x = batch[0] if isinstance(batch, tuple) else batch

        z = self._eval_encoder(x)
        x_hat = self._eval_decoder(z)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return {"loss": loss.item()}

    def _optimizer(self):
        return optim.Adam(self.parameters(), lr=1e-3)

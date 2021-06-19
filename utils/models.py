import torch.nn as nn
from torch import optim
import torch


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=256, kernel_size=(7, 7), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, X):
        X = self.generator(X)
        return X


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 2), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 1),

            nn.Sigmoid()
        )

    def forward(self, X):
        X = self.classifier.forward(X)
        return X


class GAN(object):
    def __init__(self, generator, discriminator, device, learning_rate=.0001, loss=nn.BCELoss()):
        self.real_label, self.fake_label = 1., 0.
        self.device = device

        # Models
        self.discriminator = discriminator
        self.generator = generator

        # Optimizers & Loss Function
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.loss = loss

        # Fixed noise to track progress
        self.fixed_noise = torch.rand((32, 100, 1, 1), device=self.device)

    def train(self, num_epochs=100, verbose=True):

        pass

    def generate(self):
        pass

    def save(self, location):
        pass

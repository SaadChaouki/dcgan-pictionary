import torch.nn as nn
from torch import optim
import torch
import warnings

warnings.filterwarnings("ignore")


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

    def generate(self, device='cpu'):
        noise_input = torch.rand((10, 100, 1, 1), device=device)
        return self.forward(noise_input)


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

        # Models - Discriminator and Generator
        self.discriminator = discriminator
        self.generator = generator

        # Optimizers & Loss Function
        self.optimizer_discriminator = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.optimizer_generator = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.loss = loss

        # Fixed noise to track progress
        self.fixed_noise = torch.rand((32, 100, 1, 1), device=self.device)

    def train(self, training_data, num_epochs=200, verbose=True, verbose_step=50):
        # Check data loader is correct
        assert isinstance(training_data, torch.utils.data.dataloader.DataLoader)

        # Lists to track the array
        generator_loss, discriminator_loss, fixed_predictions = [], [], []

        # Starting the epochs training
        print(f'Training Discriminator and Generator on {num_epochs} epochs.')

        # Training
        for epoch in range(num_epochs):
            for i, data_batch in enumerate(training_data):
                # Capture batch size
                batch_size = data_batch.size(0)

                # Training with the real data. Set the real data to the device used and create labels.
                self.discriminator.zero_grad()
                real_data = data_batch.to(self.device)
                label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)

                # Predict on the real data
                model_output = self.discriminator.forward(real_data).view(-1)

                # Compute error on the real labels
                error_discriminator_real = self.loss(model_output, label)
                error_discriminator_real.backward()
                discriminator_real_avg_prediction = model_output.mean().item()

                # Training with the fake data. Moving to the fake data and following the same steps.
                noise_input = torch.rand((batch_size, 100, 1, 1), device=self.device)
                fake_data = self.generator.forward(noise_input)
                label.fill_(self.fake_label)

                # Predict on the fake data
                model_output = self.discriminator.forward(fake_data.detach()).view(-1)

                # Compute error on the fake labels
                error_discriminator_fake = self.loss(model_output, label)
                error_discriminator_fake.backward()
                discriminator_fake_avg_prediction_one = model_output.mean().item()

                # Compute total error and step
                error_discriminator = error_discriminator_fake + error_discriminator_real
                self.optimizer_discriminator.step()

                # Generator training - Predict on the fake data and flag it as real for the generator
                self.generator.zero_grad()
                label.fill_(self.real_label)
                model_output = self.discriminator.forward(fake_data).view(-1)

                # Compute generator error
                error_generator = self.loss(model_output, label)
                error_generator.backward()
                discriminator_fake_avg_prediction_two = model_output.mean().item()

                # Step for the generator
                self.optimizer_generator.step()

                # Track the error
                generator_loss.append(error_generator.item())
                discriminator_loss.append(error_discriminator.item())

                if verbose:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch + 1, num_epochs, i + 1, len(training_data),
                             error_discriminator.item(), error_generator.item(), error_discriminator_real,
                             discriminator_fake_avg_prediction_one, discriminator_fake_avg_prediction_two))

    def save(self, location):
        torch.save(self.generator.state_dict(), f'{location}_generator.pt')
        torch.save(self.discriminator.state_dict(), f'{location}_discriminator.pt')

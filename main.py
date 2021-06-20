import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np

from utils.processing import DataManager, QuickDraw, Rescale, Normalize
from utils.models import Discriminator, Generator, GAN
from utils.helpers import display_image

if __name__ == '__main__':
    # Set the name of the dataset
    drawing_name = 'star.npy'

    # Parameters
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    data_limit = 50000
    image_size = 28

    # Download the data if it does not exist.
    data_manager = DataManager()
    data_manager.download(drawing_name)
    raw_data = data_manager.load(drawing_name, data_limit=data_limit)

    # Reshaping data to 3D array of size 1x28x28
    reshaped_data = raw_data.reshape((len(raw_data), 1, image_size, image_size))
    del raw_data

    # Creating the Quick, Draw! torch dataset
    quick_draw = QuickDraw(reshaped_data, transform=transforms.Compose([Rescale(255), Normalize()]))
    del reshaped_data

    # Data Loader
    data_loader = torch.utils.data.DataLoader(quick_draw, batch_size=64, shuffle=True)

    # Models - Discriminator and Generator
    discriminator = Discriminator()
    generator = Generator()

    # Create GAN model
    gan = GAN(generator=generator, discriminator=discriminator, device=device)

    # Train model
    gan.train(data_loader, num_epochs=10)

    # Save the trained model
    gan.save(f'models/star/')



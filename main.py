import torchvision.transforms as transforms
import torch

from utils.processing import DataManager, QuickDraw, Rescale, Normalize
from utils.models import Discriminator, Generator, GAN

if __name__ == '__main__':
    # Parameters
    parameters = {
        'dataset': 'star',
        'device': torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu"),
        'data_limit': 10000,
        'image_size': 28,
        'batch_size': 64,
        'num_epochs': 5,
        'learning_rate': .001
    }

    # Set the name of the dataset
    drawing_name = f'{parameters["dataset"]}.npy'

    # Download the data if it does not exist.
    data_manager = DataManager()
    data_manager.download(drawing_name)
    raw_data = data_manager.load(drawing_name, data_limit=parameters['data_limit'])

    # Reshaping data to 3D array of size 1x28x28
    reshaped_data = raw_data.reshape((len(raw_data), 1, parameters['image_size'], parameters['image_size']))
    del raw_data

    # Creating the Quick, Draw! torch dataset
    quick_draw = QuickDraw(reshaped_data, transform=transforms.Compose([Rescale(255), Normalize()]))
    del reshaped_data

    # Data Loader
    data_loader = torch.utils.data.DataLoader(quick_draw, batch_size=parameters['batch_size'], shuffle=True)

    # Models - Discriminator and Generator
    discriminator = Discriminator()
    generator = Generator()

    # Set the weights


    # Create GAN model
    gan = GAN(generator=generator, discriminator=discriminator,
              device=parameters['device'], learning_rate=parameters['learning_rate'])

    # Train model
    gan.train(data_loader, num_epochs=parameters['num_epochs'])

    # Save the trained model
    gan.save(f'models/{parameters["dataset"]}/')



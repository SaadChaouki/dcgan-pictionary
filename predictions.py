from utils.models import Generator
import torch
from utils.helpers import display_image


if __name__ == '__main__':
    # Data
    dataset = 'star'

    # Load the trained model
    generator = Generator()
    generator.load_state_dict(torch.load(f'models/{dataset}/generator.pt'))
    generator.eval()

    # Generate image
    image = generator.generate().detach().numpy()

    # Draw
    display_image(image[0][0], reshape=False)



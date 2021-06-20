from utils.models import Generator
import torch
from utils.helpers import display_image


if __name__ == '__main__':

    # Load the trained model
    generator = Generator()
    generator.load_state_dict(torch.load('models/star_generator.pt'))
    generator.eval()

    # Generate image
    image = generator.generate().detach().numpy()

    # Draw
    display_image(image[1][0], reshape=False)



import torchvision.transforms as transforms
import numpy as np

from utils.processing import DataManager, QuickDraw, Rescale, Normalize
from utils.helpers import display_image

if __name__ == '__main__':
    # Set the name of the dataset
    drawing_name = 'zigzag.npy'

    # Parameters
    data_limit = 200
    image_size = 28

    # Download the data if it does not exist.
    data_manager = DataManager()
    data_manager.download(drawing_name)
    raw_data = data_manager.load(drawing_name, data_limit=data_limit)

    # Reshaping data to 2D array of size 28x28
    reshaped_data = raw_data.reshape((len(raw_data), image_size, image_size))
    del raw_data

    # Creating the Quick, Draw! torch dataset
    quick_draw = QuickDraw(reshaped_data, transform=transforms.Compose([Rescale(255), Normalize()]))
    del reshaped_data


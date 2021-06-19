import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("TkAgg")


def display_image(array, size=28, title='Image Display', reshape=True):
    """
    function to display a single image.
    """
    image_array = np.reshape(array, (size, size)) if reshape else array
    plt.clf()
    plt.imshow(image_array, cmap='Greys', aspect='auto')
    plt.title(title)
    plt.axis(False)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import data

def convolve_image(kernel):
    image = data.camera()
    convolved_image = signal.convolve2d(image, kernel, mode="same", boundary="symm")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(convolved_image, cmap="gray")
    plt.title("Convolved Image")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    convolve_image(kernel)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import data


def convolve_image(kernel):
    # Read the input image
    image = data.camera()  # You can replace this with your own image data

    # Convolve the image with the kernel using scipy's convolve2d
    convolved_image = signal.convolve2d(image, kernel, mode="same", boundary="symm")

    # Plot the original and convolved image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(convolved_image, cmap="gray")
    plt.title("Convolved Image")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create a simple 3x3 kernel
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # Convolve the image with the kernel
    convolve_image(kernel)

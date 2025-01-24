import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from imageio.v2 import imread
import cv2


def rgb_to_grayscale(image):
    # Check if the image is RGB (3 channels)
    if image.ndim == 3:
        # Convert to grayscale using the standard luminance formula
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image  # If already grayscale, return as is


def convolve_image(input_image_path, kernel):
    # Load the image
    # image = imread(input_image_path)
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Debug: Check the image shape
    print("Image shape before conversion:", image.shape)

    # Convert to grayscale if necessary
    # image = rgb_to_grayscale(image)

    # Debug: Check the image shape after conversion
    print("Image shape after conversion:", image.shape)

    # Convolve the image with the kernel
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
    # Define the kernel
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # Input image path
    input_image_path = "banner.png"  # Replace with your image path

    # Perform convolution
    convolve_image(input_image_path, kernel)

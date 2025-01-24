import cv2
import numpy as np
from matplotlib import pyplot as plt


def apply_unsharp_mask(image_path, sigma=1.5, strength=1.5):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(grayscale_image, (0, 0), sigma)

    # Calculate the unsharp mask
    unsharp_mask = grayscale_image - blurred_image

    # Enhance the edges with the unsharp mask
    sharpened_image = grayscale_image + strength * unsharp_mask

    # Clip values to the valid range [0, 255]
    sharpened_image = np.clip(sharpened_image, 0, 255)

    # Display the original and sharpened images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale_image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image, cmap="gray")
    plt.title("Sharpened Image")
    plt.show()
    return sharpened_image


# Apply unsharp masking
image_path = "banner.png"
sharpened_image = apply_unsharp_mask(image_path, sigma=1.5, strength=1.5)

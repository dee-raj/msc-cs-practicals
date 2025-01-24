import cv2
import numpy as np
from matplotlib import pyplot as plt


def unsharp_masking(image_path, sigma=1.5, strength=1.5):
    """
    Apply smoothing, sharpening, and unsharp masking to enhance an image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for smoothing
    blurred_image = cv2.GaussianBlur(grayscale_image, (0, 0), sigma)

    # Create a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply the sharpening kernel using 2D convolution
    sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)

    # Enhance the edges with the unsharp mask formula
    unsharp_image = grayscale_image + strength * (grayscale_image - blurred_image)

    # Clip values to the valid range [0, 255]
    unsharp_image = np.clip(unsharp_image, 0, 255).astype(np.uint8)

    # Display the results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(grayscale_image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_image, cmap="gray")
    plt.title("Blurred Image (Smoothing)")

    plt.subplot(1, 3, 3)
    plt.imshow(unsharp_image, cmap="gray")
    plt.title("Sharpened Image (Unsharp Masking)")
    plt.show()

    return unsharp_image


# Path to the input image
image_path = "banner.png"

# Apply unsharp masking with specified parameters
sharpened_image = unsharp_masking(image_path, sigma=1.5, strength=1.5)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_sobel(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator to find edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize the gradient magnitude to the range [0, 255]
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    return gradient_magnitude


def apply_canny(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def main():
    # Load the image
    image_path = "banner.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image from {image_path}")
        return

    # Apply Sobel edge detection
    sobel_result = apply_sobel(image)

    # Apply Canny edge detection
    canny_result = apply_canny(image)

    # Display the original image and the results
    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(132)
    plt.imshow(sobel_result, cmap="gray")
    plt.title("Sobel Edge Detection")

    plt.subplot(133)
    plt.imshow(canny_result, cmap="gray")
    plt.title("Canny Edge Detection")

    plt.show()


if __name__ == "__main__":
    main()

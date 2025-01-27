import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    return gradient_magnitude

def apply_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

if __name__ == "__main__":
    image_path = "banner.png"
    image = cv2.imread(image_path)

    sobel_result = apply_sobel(image)
    canny_result = apply_canny(image)

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

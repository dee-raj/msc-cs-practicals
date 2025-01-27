import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gradient_laplacian(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = np.uint8(
        255 * gradient_magnitude / np.max(gradient_magnitude)
    )

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(255 * laplacian / np.max(np.abs(laplacian)))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(gradient_magnitude, cmap="gray")
    plt.title("Gradient Magnitude")

    plt.subplot(1, 3, 3)
    plt.imshow(laplacian, cmap="gray")
    plt.title("Laplacian Image")
    plt.show()

    cv2.imshow("Gradient Magnitude", gradient_magnitude)
    cv2.imshow("Laplacian Image", laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "banner.png"
apply_gradient_laplacian(image_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply histogram equalization to improve the contrast
    equalized_img = cv2.equalizeHist(img)

    # Plot the original and equalized images
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")

    # Equalized Image
    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img, cmap="gray")
    plt.title("Equalized Image")

    plt.show()

    # Optionally save the equalized image
    # cv2.imwrite("equalized_image.jpg", equalized_img)

    # Display the equalized image
    cv2.imshow("Equalized Image", equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "banner.png"
histogram_equalization(image_path)

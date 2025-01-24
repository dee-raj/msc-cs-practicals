import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_threshold(image_path, threshold_value):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to convert the image to binary (black and white)
    _, thresholded_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    # Plot the original and thresholded images side by side
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")

    # Thresholded Image
    plt.subplot(1, 2, 2)
    plt.imshow(thresholded_img, cmap="gray")
    plt.title("Thresholded Image")

    plt.show()

    # Optionally save the thresholded image
    # cv2.imwrite("thresholded_image.jpg", thresholded_img)

    # Display the thresholded image in a new window
    cv2.imshow("Thresholded Image", thresholded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = "banner.png"
threshold_value = 127  # You can adjust this value to get different results
apply_threshold(image_path, threshold_value)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_morphological_operations(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Display the original image
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    # Erosion
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)

    # Display the eroded image
    plt.subplot(132)
    plt.imshow(erosion, cmap="gray")
    plt.title("Erosion")

    # Dilation
    dilation = cv2.dilate(image, kernel, iterations=1)

    # Display the dilated image
    plt.subplot(133)
    plt.imshow(dilation, cmap="gray")
    plt.title("Dilation")

    plt.show()

    # Opening (erosion followed by dilation)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Closing (dilation followed by erosion)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Display the images after opening and closing
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(opening, cmap="gray")
    plt.title("Opening")

    plt.subplot(122)
    plt.imshow(closing, cmap="gray")
    plt.title("Closing")

    plt.show()


def main():
    image_path = "banner.png"
    apply_morphological_operations(image_path)


if __name__ == "__main__":
    main()

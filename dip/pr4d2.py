import cv2
import numpy as np
import matplotlib.pyplot as plt


def floyd_steinberg_dithering(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = np.float32(img)
    height, width = img.shape

    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            img[y, x] = new_pixel

            quant_error = old_pixel - new_pixel
            img[y, x + 1] += quant_error * 7 / 16
            img[y + 1, x - 1] += quant_error * 3 / 16
            img[y + 1, x] += quant_error * 5 / 16
            img[y + 1, x + 1] += quant_error * 1 / 16

    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap="gray")
    plt.title("Halftone Image")
    plt.show()

    # cv2.imwrite('halftone_image.jpg', img)
    cv2.imshow("Halftone Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "banner.png"
floyd_steinberg_dithering(image_path)

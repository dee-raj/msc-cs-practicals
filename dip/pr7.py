import cv2
import numpy as np
from matplotlib import pyplot as plt

sigma = 1.5
strength = 1.5
image_path = "banner.png"
image = cv2.imread(image_path)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(grayscale_image, (0, 0), sigma)

sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)

unsharp_image = grayscale_image + strength * (grayscale_image - blurred_image)
unsharp_image = np.clip(unsharp_image, 0, 255).astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(2, 3, 1)
plt.imshow(grayscale_image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 3, 2)
plt.imshow(blurred_image, cmap="gray")
plt.title("Blurred Image (Smoothing)")

plt.subplot(2, 3, 3)
plt.imshow(sharpened_image, cmap="gray")
plt.title("Sharpened Image (Sharpening)")

plt.subplot(2, 3, 4)
plt.imshow(unsharp_image, cmap="gray")
plt.title("Unsharp masking Image (Unsharp Masking)")
plt.show()

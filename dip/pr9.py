import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "banner.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

erosion = cv2.erode(image, kernel, iterations=1)
plt.subplot(132)
plt.imshow(erosion, cmap="gray")
plt.title("Erosion")


dilation = cv2.dilate(image, kernel, iterations=1)
plt.subplot(133)
plt.imshow(dilation, cmap="gray")
plt.title("Dilation")
plt.show()

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(opening, cmap="gray")
plt.title("Opening")

plt.subplot(122)
plt.imshow(closing, cmap="gray")
plt.title("Closing")
plt.show()
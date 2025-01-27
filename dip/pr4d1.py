import cv2
import matplotlib.pyplot as plt

def apply_threshold(image_path, threshold_value):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(thresholded_img, cmap="gray")
    plt.title("Thresholded Image")
    plt.show()

    # cv2.imwrite("thresholded_image.jpg", thresholded_img)
    cv2.imshow("Thresholded Image", thresholded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "banner.png"
threshold_value = 127
apply_threshold(image_path, threshold_value)

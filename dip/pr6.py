import cv2

image = cv2.imread("banner.png")
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1)
median_blur = cv2.medianBlur(image, 5)

cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Blur (Linear Smoothing)", gaussian_blur)
cv2.imshow("Median Blur (Nonlinear Smoothing)", median_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Read the image
image = cv2.imread("banner.png")

# Apply Gaussian Blur (Linear Smoothing)
gaussian_blur = cv2.GaussianBlur(
    image, (5, 5), 1
)  # Kernel size and sigma can be adjusted

# Apply Median Blur (Nonlinear Smoothing)
median_blur = cv2.medianBlur(image, 5)  # Kernel size must be odd (e.g., 3, 5, 7)

# Display the images
cv2.imshow("Original Image", image)
cv2.imshow("Gaussian Blur (Linear Smoothing)", gaussian_blur)
cv2.imshow("Median Blur (Nonlinear Smoothing)", median_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()

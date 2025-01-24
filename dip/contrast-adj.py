import cv2

# Read the input image
image = cv2.imread("banner.png")

# Define the alpha (contrast) and beta (brightness)
alpha = 1.5  # Contrast control (1.0 means no change, higher values increase contrast)
beta = 10  # Brightness control (0 means no change, positive values increase brightness)

# Apply contrast and brightness adjustments using the specified alpha and beta values
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Display the original image and the adjusted image
cv2.imshow("Original Image", image)
cv2.imshow("Adjusted Image", adjusted)

# Wait for any key press to close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# Load an image
image = cv2.imread("banner.png", cv2.IMREAD_COLOR)

# Display the original image
window_name = "Original Image"
cv2.imshow("Original Image", image)
cv2.waitKey(0)


# Perform downsampling
scale_factor = 0.5  # For example, to downsample by 2
downsampled_image = cv2.resize(
    image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
)

# Display the downsampled image
cv2.imshow("Downsampled Image", downsampled_image)
cv2.waitKey(0)


# Perform upsampling
scale_factor = 2  # For example, to upsample by 2
upsampled_image = cv2.resize(
    image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
)

# Display the upsampled image
cv2.imshow("Upsampled Image", upsampled_image)
cv2.waitKey(0)

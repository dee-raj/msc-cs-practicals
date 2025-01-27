import cv2

image = cv2.imread("banner.png", cv2.IMREAD_COLOR)
window_name = "Original Image"
cv2.imshow("Original Image", image)
cv2.waitKey(0)

scale_factor = 0.5
downsampled_image = cv2.resize(
    image, None, fx=scale_factor, fy=scale_factor,
    interpolation=cv2.INTER_LINEAR
)

cv2.imshow("Downsampled Image", downsampled_image)
cv2.waitKey(0)

scale_factor = 2
upsampled_image = cv2.resize(
    image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR
)

cv2.imshow("Upsampled Image", upsampled_image)
cv2.waitKey(0)

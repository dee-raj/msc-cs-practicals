import cv2

image = cv2.imread("banner.png")
alpha = 1.5
beta = 10
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow("Original Image", image)
cv2.imshow("Adjusted Image", adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()

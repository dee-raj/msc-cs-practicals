import cv2
import numpy as np

img = cv2.imread("banner.png")

img_log = (np.log(img + 1e-4) / (np.log(1 + np.max(img)))) * 255
img_log = np.array(img_log, dtype=np.uint8)

cv2.imshow("Log Transformed Image", img_log)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

for gamma in [0.1, 0.5, 1.2, 2.2]:
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype="uint8")

    cv2.imshow(f"Gamma Transformed (γ={gamma})", gamma_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

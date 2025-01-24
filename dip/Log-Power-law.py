import cv2
import numpy as np

# Load the image
img = cv2.imread("banner.png")

# Apply Log Transformation
# To avoid taking the log of zero, add a small constant (1e-4) to the image
img_log = (np.log(img + 1e-4) / (np.log(1 + np.max(img)))) * 255

# Convert the transformed image to uint8 for display
img_log = np.array(img_log, dtype=np.uint8)

# Display the original image and the log-transformed image
cv2.imshow("Log Transformed Image", img_log)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Gamma Correction with 4 different gamma values
for gamma in [0.1, 0.5, 1.2, 2.2]:
    # Normalize the image and apply gamma correction
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype="uint8")

    # Display the gamma-corrected image
    cv2.imshow(f"Gamma Transformed (γ={gamma})", gamma_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

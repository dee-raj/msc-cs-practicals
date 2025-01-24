import cv2
import numpy as np
import matplotlib.pyplot as plt


def log_transform(image):
    # Normalize the image to [0, 1]
    normalized_img = image / 255.0
    # Apply the log transformation
    log_image = (np.log1p(normalized_img) / np.log(1 + np.max(normalized_img))) * 255
    # Convert back to uint8
    return np.array(log_image, dtype=np.uint8)


def gamma_transform(image, gamma):
    # Normalize the image to [0, 1]
    normalized_img = image / 255.0
    # Apply the gamma correction
    gamma_image = np.power(normalized_img, gamma) * 255
    # Convert back to uint8
    return np.array(gamma_image, dtype=np.uint8)


def main():
    # Load the input image
    img = cv2.imread("banner.png", cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Apply log transformation
    img_log = log_transform(img)

    # Apply gamma transformations for different values of gamma
    gamma_values = [0.5, 1.0, 2.0]
    gamma_images = [gamma_transform(img, gamma) for gamma in gamma_values]

    # Plot the results using a 2x3 grid
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Log-transformed image
    plt.subplot(2, 3, 2)
    plt.imshow(img_log, cmap="gray")
    plt.title("Log Transformation")
    plt.axis("off")

    # Gamma-transformed images
    for i, (gamma, gamma_img) in enumerate(zip(gamma_values, gamma_images)):
        plt.subplot(2, 3, 3 + i)
        plt.imshow(gamma_img, cmap="gray")
        plt.title(f"Gamma (γ={gamma})")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

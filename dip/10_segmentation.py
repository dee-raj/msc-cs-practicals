import cv2
import numpy as np
import matplotlib.pyplot as plt


# Harris Corner Detection
def harris_corner_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Harris Corner Detector
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Dilate corners to mark them on the original image
    corners = cv2.dilate(corners, None)

    # Create a copy of the original image to overlay detected corners
    result_image = image.copy()
    result_image[corners > 0.01 * corners.max()] = [0, 0, 255]

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corner Detection")
    plt.show()


# Blob Detection
def blob_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up blob detector
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(gray)

    # Draw detected blobs on the image
    blob_image = cv2.drawKeypoints(
        image,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
    plt.title("Blob Detection")
    plt.show()


# Histogram of Oriented Gradients (HoG)
def hog_feature_extraction(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize HoG descriptor
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)

    print(f"HoG Features Shape: {features.shape}")

    # Display the image and features
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("HoG Feature Extraction")
    plt.show()


# Haar Cascade Features
def haar_feature_detection(
    image_path, cascade_path="haarcascade_frontalface_default.xml"
):
    # Load the Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    # Read and convert the image to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    result_image = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Haar Cascade Detection")
    plt.show()


# Main function
def main():
    image_path = "banner.png"

    # Apply Harris Corner Detection
    print("Harris Corner Detection:")
    harris_corner_detection(image_path)

    # Apply Blob Detection
    print("Blob Detection:")
    blob_detection(image_path)

    # Apply HoG Feature Extraction
    print("HoG Feature Extraction:")
    hog_feature_extraction(image_path)

    # Apply Haar Feature Detection
    print("Haar Feature Detection:")
    haar_feature_detection(image_path)


if __name__ == "__main__":
    main()

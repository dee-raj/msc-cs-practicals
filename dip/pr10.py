import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    result_image = image.copy()
    result_image[corners > 0.01 * corners.max()] = [0, 0, 255]

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Harris Corner Detection")
    plt.show()


def blob_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(gray)

    blob_image = cv2.drawKeypoints(
        image,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB))
    plt.title("Blob Detection")
    plt.show()


def hog_feature_extraction(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)
    print(f"HoG Features Shape: {features.shape}")

    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("HoG Feature Extraction")
    plt.show()


def haar_feature_detection(
    image_path, cascade_path="haarcascade_frontalface_default.xml"
):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    result_image = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Haar Cascade Detection")
    plt.show()


if __name__ == "__main__":
    image_path = "banner.png"

    print("Harris Corner Detection:")
    harris_corner_detection(image_path)

    print("Blob Detection:")
    blob_detection(image_path)

    print("HoG Feature Extraction:")
    hog_feature_extraction(image_path)

    print("Haar Feature Detection:")
    haar_feature_detection(image_path)

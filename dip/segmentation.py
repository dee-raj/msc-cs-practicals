import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function for Edge-based Segmentation
def edge_based_segmentation(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detectioSegmentationn
    edges = cv2.Canny(gray, 50, 150)

    # Display the original image and edges
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge-based Segmentation")
    plt.show()


# Function for Region-based Segmentation
def region_based_segmentation(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for region-based segmentation
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    # Display the original image and region-based segmentation result
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(thresh, cmap="gray")
    plt.title("Region-based Segmentation")
    plt.show()


# Function for Line Detection
def line_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the Hough Line Transform to detect lines
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )

    # Draw lines on the original image
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title("Line Detection")
    plt.show()


# Function for Circle Detection
def circle_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50,
    )

    # Draw circles on the original image
    circle_image = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(circle_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(circle_image, (i[0], i[1]), 2, (0, 255, 0), 3)

    # Display the result
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB))
    plt.title("Circle Detection")
    plt.show()


# Function for Contour-based Shape Detection
def shape_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and label shapes
    shape_image = image.copy()
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        cv2.drawContours(shape_image, [approx], -1, (0, 255, 0), 2)

        # Label shapes based on number of vertices
        x, y, w, h = cv2.boundingRect(approx)
        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) > 10:
            shape_name = "Circle"
        else:
            shape_name = "Polygon"

        cv2.putText(
            shape_image,
            shape_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Display the result
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(shape_image, cv2.COLOR_BGR2RGB))
    plt.title("Shape Detection")
    plt.show()


# Main Function
def main():
    image_path = "banner.png"  # Replace with the path to your image

    print("Applying Edge-based Segmentation...")
    edge_based_segmentation(image_path)

    print("Applying Region-based Segmentation...")
    region_based_segmentation(image_path)

    print("Detecting Lines...")
    line_detection(image_path)

    print("Detecting Circles...")
    circle_detection(image_path)

    print("Detecting Shapes...")
    shape_detection(image_path)


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_based_segmentation(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge-based Segmentation")
    plt.show()

def region_based_segmentation(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(thresh, cmap="gray")
    plt.title("Region-based Segmentation")
    plt.show()


def line_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, 
        minLineLength=50, maxLineGap=10
    )

    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title("Line Detection")
    plt.show()


def circle_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    circle_image = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(circle_image, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(circle_image, (i[0], i[1]), 2, (0, 255, 0), 3)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB))
    plt.title("Circle Detection")
    plt.show()


def shape_detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_image = image.copy()
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        cv2.drawContours(shape_image, [approx], -1, (0, 255, 0), 2)

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

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(shape_image, cv2.COLOR_BGR2RGB))
    plt.title("Shape Detection")
    plt.show()

if __name__ == "__main__":
    image_path = "source.png" 

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

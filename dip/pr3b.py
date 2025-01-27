import cv2
import matplotlib.pyplot as plt

def template_matching(source_image_path, template_image_path):
    source_image = cv2.imread(source_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    template_height, template_width = template_image.shape
    result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc 
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

    result_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(source_image, cmap="gray")
    plt.title("Source Image")

    plt.subplot(1, 3, 2)
    plt.imshow(template_image, cmap="gray")
    plt.title("Template Image")

    plt.subplot(1, 3, 3)
    plt.imshow(result_image, cmap="gray")
    plt.title("Detected Match")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    source_image_path = "source.png"
    template_image_path = "template.png"

    template_matching(source_image_path, template_image_path)

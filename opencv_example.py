import cv2
import numpy as np


def main():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (200, 120, 50)

    cv2.rectangle(image, (40, 40), (280, 180), (255, 255, 255), thickness=2)
    cv2.circle(image, (160, 120), 40, (0, 255, 0), thickness=3)
    cv2.putText(
        image,
        "OpenCV",
        (90, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Convert to grayscale and detect edges.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    cv2.imwrite("opencv_example_output.png", image)
    cv2.imwrite("opencv_example_edges.png", edges)

    print("Created opencv_example_output.png")
    print("Created opencv_example_edges.png")


if __name__ == "__main__":
    main()

import cv2
import numpy as np


def create_synthetic_image():
    image = np.full((360, 480, 3), fill_value=240, dtype=np.uint8)

    cv2.rectangle(image, (50, 50), (180, 180), (0, 128, 255), -1)
    cv2.circle(image, (320, 120), 60, (0, 255, 0), -1)
    points = np.array([[260, 260], [190, 340], [330, 340]], np.int32)
    cv2.fillPoly(image, [points], (255, 0, 0))

    return image


def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 3:
        return "triangle"
    if len(approx) == 4:
        return "rectangle"
    return "circle"


def detect_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        label = classify_shape(contour)
        detections.append({"label": label, "box": (x, y, w, h)})

    return detections


def draw_detections(image, detections):
    output = image.copy()
    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["box"]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(
            output,
            f"{idx}: {det['label']}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def main():
    image = create_synthetic_image()
    detections = detect_objects(image)

    print("Detected objects:")
    for det in detections:
        print(f"  - {det['label']} at {det['box']}")

    output = draw_detections(image, detections)
    cv2.imwrite("object_detection_example_output.png", output)
    print("Saved object_detection_example_output.png")


if __name__ == "__main__":
    main()

DUE:
pip install pyttsx3


import sys
import os
import cv2
import math
import RPi.GPIO as GPIO
from ultralytics import YOLO

# ------------------------------------------------
# INITIALIZATION

# Pin for the button
button_pin = 17  # Assuming GPIO17 for the button

# YOLO definitions
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

desired_classes = ["person", "chair", "backpack", "handbag"]

# Define overlap thresholds
chair_overlap_threshold = 0.5
person_overlap_threshold = 0.85
bag_overlap_threshold = 0.3
bag_on_chair_overlap_threshold = 0.5

chair_confidence_threshold = 0.1

# Initialize YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ------------------------------------------------
# FUNCTIONS

def calculate_overlap(boxA, boxB):
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return overlap_area / min(boxA_area, boxB_area)


def merge_boxes(boxA, boxB):
    return (
        min(boxA[0], boxB[0]),
        min(boxA[1], boxB[1]),
        max(boxA[2], boxB[2]),
        max(boxA[3], boxB[3])
    )


def start_object_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open webcam.")
        return

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        detected_objects = []

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name in desired_classes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    detected_objects.append(
                        {"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence, "display": True})

        merged_objects = []

        for obj in detected_objects:
            is_merged = False
            if obj['class'] == 'chair' and obj['confidence'] < chair_confidence_threshold:
                continue

            for merged_obj in merged_objects:
                overlap = calculate_overlap(obj['bbox'], merged_obj['bbox'])

                if obj['class'] == 'chair' and merged_obj['class'] == 'chair' and overlap > chair_overlap_threshold:
                    merged_bbox = merge_boxes(obj['bbox'], merged_obj['bbox'])
                    merged_obj['bbox'] = merged_bbox
                    is_merged = True
                    break
                elif obj['class'] == 'person' and merged_obj['class'] == 'person' and overlap > person_overlap_threshold:
                    merged_bbox = merge_boxes(obj['bbox'], merged_obj['bbox'])
                    merged_obj['bbox'] = merged_bbox
                    is_merged = True
                    break
                elif (obj['class'] in ['backpack', 'bag'] and merged_obj['class'] in ['backpack', 'bag'] and
                      overlap > bag_overlap_threshold):
                    merged_bbox = merge_boxes(obj['bbox'], merged_obj['bbox'])
                    merged_obj['bbox'] = merged_bbox
                    is_merged = True
                    break

            if not is_merged:
                merged_objects.append(obj)

        for obj in merged_objects:
            if obj['class'] in ['backpack', 'bag']:
                for chair in merged_objects:
                    if chair['class'] == 'chair':
                        overlap = calculate_overlap(obj['bbox'], chair['bbox'])
                        if overlap > bag_on_chair_overlap_threshold:
                            chair['class'] = 'bag_on_chair'
                            obj['display'] = False
                            break

        for obj in merged_objects:
            if 'display' in obj and not obj['display']:
                continue

            bbox = obj['bbox']
            confidence = obj['confidence']
            full_label = f"{obj['class']} - {confidence * 100:.2f}%"

            if obj['class'] == 'person':
                color = (0, 255, 0)
            elif obj['class'] == 'bag_on_chair':
                color = (255, 255, 0)
            elif obj['class'] == 'chair':
                color = (255, 0, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            mid_x = int((bbox[0] + bbox[2]) / 2)
            mid_y = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(img, (mid_x, mid_y), 5, color, -1)
            cv2.putText(img, full_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Press the button to start object detection...")
    GPIO.wait_for_edge(button_pin, GPIO.FALLING)
    print("Button pressed. Starting object detection...")
    start_object_detection()


if __name__ == "__main__":
    main()

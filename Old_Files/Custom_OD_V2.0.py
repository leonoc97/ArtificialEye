from ultralytics import YOLO
import cv2
import math

# Start webcam or specify video file path
video_path = 'seat2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

desired_classes = ["person", "chair", "backpack"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates and labels
    detected_objects = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Check if the class is in the desired classes
            if class_name in desired_classes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print(f"{class_name} - Confidence: {confidence}")

                detected_objects.append({"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence})

    # Filter out overlapping bounding boxes
    filtered_objects = []
    for obj in detected_objects:
        bbox = obj["bbox"]
        overlap = False

        for other_obj in detected_objects:
            if obj != other_obj:
                other_bbox = other_obj["bbox"]

                # Calculate overlap
                overlap_area = (
                        max(0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])) *
                        max(0, min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1]))
                )

                # Calculate area of the bounding boxes
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                other_bbox_area = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])

                # Calculate overlap ratio
                overlap_ratio = overlap_area / min(bbox_area, other_bbox_area)

                # If overlap ratio is above a threshold, choose the one with higher confidence
                if overlap_ratio > 0.5:  # You can adjust this threshold
                    if obj["confidence"] >= other_obj["confidence"]:
                        overlap = True
                        break

        if not overlap:
            filtered_objects.append(obj)

    # Draw bounding boxes for persons, chairs, and backpacks
    for obj in filtered_objects:
        bbox = obj["bbox"]
        confidence = obj["confidence"]
        color = (0, 255, 0) if obj["class"] == "person" else (
        255, 0, 0)  # Green for persons, Blue for chairs and backpacks
        label = f"{obj['class']} - Confidence: {confidence * 100:.2f}%"

        # Draw bounding box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        # Calculate and draw midpoint
        mid_x = int((bbox[0] + bbox[2]) / 2)
        mid_y = int((bbox[1] + bbox[3]) / 2)
        cv2.circle(img, (mid_x, mid_y), 5, color, -1)  # Draw a small circle at the midpoint

        # Draw confidence rating
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
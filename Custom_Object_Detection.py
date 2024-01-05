from ultralytics import YOLO
import cv2
import math

# Start webcam or specify video file path
video_path = ('seat1.mp4')
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
    if not success:
        break  # Exit the loop if no frame is captured

    results = model(img, stream=True)

    # Coordinates and labels
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
                print(f"{class_name} - Confidence: {confidence}")

                detected_objects.append({"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence})

    # Check for overlapping bounding boxes
    filtered_objects = detected_objects.copy()

    for obj in detected_objects:
        if obj["class"] == "backpack":
            bbox = obj["bbox"]

            for other_obj in detected_objects:
                if other_obj["class"] == "chair":
                    other_bbox = other_obj["bbox"]
                    overlap_area = (
                        max(0, min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0])) *
                        max(0, min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1]))
                    )

                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    other_bbox_area = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])

                    overlap_ratio = overlap_area / min(bbox_area, other_bbox_area)

                    # Remove the chair if significant overlap
                    if overlap_ratio > 0.2:  # Adjust threshold as needed
                        filtered_objects.remove(other_obj)

    # Draw bounding boxes for persons, chairs, and backpacks
    for obj in filtered_objects:
        bbox = obj["bbox"]
        confidence = obj["confidence"]
        color = (0, 255, 0) if obj["class"] == "person" else (255, 0, 0)
        label = f"{obj['class']} - Confidence: {confidence * 100:.2f}%"

        # Draw bounding box
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        # Calculate and draw midpoint
        mid_x = int((bbox[0] + bbox[2]) / 2)
        mid_y = int((bbox[1] + bbox[3]) / 2)
        cv2.circle(img, (mid_x, mid_y), 5, color, -1)

        # Draw confidence rating
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
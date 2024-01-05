from ultralytics import YOLO
import cv2
import math

#TESSSSSSSSSfrSSSSSSSST

#ADFA DKAEWDF


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
        break  # Exit the loop if the video is over or there's an error

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

                detected_objects.append({"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence})

    # Sort detected objects by confidence and take the top 6
    top_objects = sorted(detected_objects, key=lambda x: x["confidence"], reverse=True)[:6]

    # Divide the top 6 objects into 'front' and 'back' based on their vertical position (y-coordinate)
    top_objects = sorted(top_objects, key=lambda x: x["bbox"][1])  # Sort by y1
    for i, obj in enumerate(top_objects):
        if i < 3:
            obj["position"] = "back"
        else:
            obj["position"] = "front"

    # Check for backpack on chair
    for obj1 in detected_objects:
        for obj2 in detected_objects:
            if obj1["class"] == "backpack" and obj2["class"] == "chair":
                backpack_bbox = obj1["bbox"]
                chair_bbox = obj2["bbox"]

                intersect_area = (
                    max(0, min(backpack_bbox[2], chair_bbox[2]) - max(backpack_bbox[0], chair_bbox[0])) *
                    max(0, min(backpack_bbox[3], chair_bbox[3]) - max(backpack_bbox[1], chair_bbox[1]))
                )
                backpack_area = (backpack_bbox[2] - backpack_bbox[0]) * (backpack_bbox[3] - backpack_bbox[1])

                if intersect_area / backpack_area > 0.9:
                    obj1["label"] = "backpackchair"
                    obj1["bbox"] = chair_bbox
                    detected_objects.remove(obj2)

    # Draw bounding boxes
    for obj in detected_objects:
        bbox = obj["bbox"]
        label = obj.get("label", obj["class"])
        position_label = obj.get("position", "")

        if label == "backpackchair":
            color = (0, 255, 255)  # Different color for backpack on chair
        else:
            color = (0, 255, 0) if obj["class"] == "person" else (255, 0, 0)

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        label_text = f"{label} - {position_label}"
        cv2.putText(img, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

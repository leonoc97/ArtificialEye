from ultralytics import YOLO
import cv2
import math

def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def is_inside(boxA, boxB):
    # Check if boxA is inside boxB
    return boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and boxA[2] <= boxB[2] and boxA[3] <= boxB[3]

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

def calculate_overlap(boxA, boxB):
    # Calculate the overlapping area between two boxes
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

while True:
    success, img = cap.read()
    if not success:
        break

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

                detected_objects.append({"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence, "display": True})

    # Merge overlapping chairs and check backpack inside chair
    merged_objects = []
    for obj in detected_objects:
        if obj['class'] == 'chair':
            # Check for overlapping chairs
            is_merged = False
            for merged_obj in merged_objects:
                if merged_obj['class'] == 'chair':
                    if calculate_overlap(obj['bbox'], merged_obj['bbox']) > 0.8:  # 80% overlap threshold
                        # Merge the bounding boxes
                        merged_x1 = min(obj['bbox'][0], merged_obj['bbox'][0])
                        merged_y1 = min(obj['bbox'][1], merged_obj['bbox'][1])
                        merged_x2 = max(obj['bbox'][2], merged_obj['bbox'][2])
                        merged_y2 = max(obj['bbox'][3], merged_obj['bbox'][3])
                        merged_obj['bbox'] = (merged_x1, merged_y1, merged_x2, merged_y2)
                        is_merged = True
                        break
            if not is_merged:
                merged_objects.append(obj)
        else:
            merged_objects.append(obj)

    # Check for backpack inside chair
    for obj in merged_objects:
        if obj['class'] == 'backpack':
            for chair in merged_objects:
                if chair['class'] == 'chair':
                    if calculate_overlap(obj['bbox'], chair['bbox']) > 0.8:  # 80% inside threshold
                        chair['class'] = 'backpack_on_chair'
                        obj['display'] = False  # Do not display the individual backpack
                        break

    # Sort the objects based on the vertical position of their midpoints
    merged_objects = sorted(merged_objects, key=lambda obj: (obj['bbox'][1] + obj['bbox'][3]) / 2, reverse=True)

    # Assign 'front' or 'back' labels
    for i, obj in enumerate(merged_objects):
        if i < 3:  # First three are closest to the bottom
            obj['position'] = 'front'
        else:
            obj['position'] = 'back'

    # Draw bounding boxes
    for obj in merged_objects:
        if obj['display']:
            bbox = obj['bbox']
            confidence = obj['confidence']
            position_label = obj.get('position', '')

            label = f"{obj['class']} ({position_label}) - Confidence: {confidence * 100:.2f}%"

            # Set color based on class
            if obj['class'] == 'person':
                color = (0, 255, 0)
            elif obj['class'] == 'backpack_on_chair':
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)

            # Draw bounding box, midpoint, and label
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            mid_x = int((bbox[0] + bbox[2]) / 2)
            mid_y = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(img, (mid_x, mid_y), 5, color, -1)
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the frame
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
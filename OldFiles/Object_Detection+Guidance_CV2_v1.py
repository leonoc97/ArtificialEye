from ultralytics import YOLO
import cv2
import math
import time

frame_width = 640
frame_height = 480

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Model
model = YOLO("../yolo-Weights/yolov8n.pt")

# Object classes
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

# Grid dimensions
grid_rows = 3
grid_cols = 3

# Define colors for grid lines and text
grid_color = (0, 255, 0)  # Green
text_color = (255, 255, 255)  # White

# Define the list of objects you want to display bounding boxes for
objects_to_display = ["cell phone", "banana", "apple", "toothbrush", "bottle", "cup"]

# Step 1: Object detection until one of the objects_to_display is detected with confidence > 85%
object_to_find = None
confidence_threshold = 0.60

# Variables for capturing an image
captured_image = None
capture_image_threshold = 0.50

# Initialize variables for smoothing
prev_box = None
smoothing_factor = 0.5  # Adjust this factor based on your preference

while object_to_find is None:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = box.conf[0]
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in objects_to_display and confidence > confidence_threshold:
                object_to_find = class_name

                # Capture image when confidence is higher than the threshold
                if confidence > capture_image_threshold:
                    captured_image = img.copy()

                break

# Display the captured image in a separate window
if captured_image is not None:
    cv2.imshow('Captured Image', captured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Step 2: Guide the bounding box of the "object to be found" to the center of the frame within the 3 by 3 grid
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Draw grid lines and labels
    cell_width = frame_width // grid_cols
    cell_height = frame_height // grid_rows

    for i in range(grid_rows):
        for j in range(grid_cols):
            # Calculate cell coordinates
            x1 = j * cell_width
            x2 = (j + 1) * cell_width
            y1 = i * cell_height
            y2 = (i + 1) * cell_height

            # Draw grid cell
            cv2.rectangle(img, (x1, y1), (x2, y2), grid_color, 1)

            # Add labels (letters for columns, numbers for rows)
            label = f"{chr(65 + j)}{i + 1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_x = (x1 + x2 - label_size[0]) // 2
            label_y = (y1 + y2 + label_size[1]) // 2
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Iterate over detected objects
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence and class
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Check if the detected object is the "object to be found"
            if class_name == object_to_find:
                # Smoothing: Update current box using a weighted average with the previous box
                if prev_box is not None:
                    x1 = int((1 - smoothing_factor) * x1 + smoothing_factor * prev_box[0])
                    y1 = int((1 - smoothing_factor) * y1 + smoothing_factor * prev_box[1])
                    x2 = int((1 - smoothing_factor) * x2 + smoothing_factor * prev_box[2])
                    y2 = int((1 - smoothing_factor) * y2 + smoothing_factor * prev_box[3])

                prev_box = (x1, y1, x2, y2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Calculate grid cell
                cell_col = int((x1 + x2) / 2 // cell_width)
                cell_row = int((y1 + y2) / 2 // cell_height)

                # Print object information
                text = f"{class_name} in cell {chr(65 + cell_col)}{cell_row + 1} with confidence: {confidence}"

                # Calculate command to guide bounding box to the center cell
                command = ""
                if cell_col < grid_cols // 2:
                    command += "Move Right "
                elif cell_col > grid_cols // 2:
                    command += "Move Left "

                if cell_row < grid_rows // 2:
                    command += "Move Down "
                elif cell_row > grid_rows // 2:
                    command += "Move Up "

                # Display the command at the bottom left of the bounding box
                cv2.putText(img, command, (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                # Display additional text with object type and confidence
                obj_info = f"{class_name} ({confidence})"
                cv2.putText(img, obj_info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # Display the image
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

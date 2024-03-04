from ultralytics import YOLO
import cv2
import math
from mqtt_publish import publish_top_configuration
import paho.mqtt.client as mqtt
import time
#from screenshot_util import take_screenshot_with_boxes

# Initialize flag to track if screenshot has been taken
screenshot_taken = False

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
screenshot_taken = False
def merge_boxes(boxA, boxB):
    return (
        min(boxA[0], boxB[0]),  # x1
        min(boxA[1], boxB[1]),  # y1
        max(boxA[2], boxB[2]),  # x2
        max(boxA[3], boxB[3])   # y2
    )


def take_screenshot_with_boxes(frame, objects, output_path):
    # Draw bounding boxes on the frame
    for obj in objects:
        # Extract the bbox coordinates from the object
        bbox = obj['bbox']  # This assumes that 'bbox' holds the coordinates
        x1, y1, x2, y2 = bbox  # Now this unpacks the coordinates directly

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write the frame with bounding boxes to an image file
    cv2.imwrite(output_path, frame)


# Start webcam or specify video file path
video_path = ('seat5.mp4')
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

desired_classes = ["person", "chair","backpack", "handbag"]

# Define overlap thresholds
chair_overlap_threshold = 0.5  # 50% overlap threshold for chairs
person_overlap_threshold = 0.85  # 85% overlap threshold for persons
bag_overlap_threshold = 0.3
bag_on_chair_overlap_threshold = 0.3  # 60% overlap threshold for bag on chair
chair_confidence_threshold = 0.1  # 10%

# Initialize configuration ID counter outside the loop
config_id_counter = 1

# Initialize a set to store encountered configurations
encountered_configurations = set()

# Outside your main loop
last_configurations = []
config_log = {}  # Define config_log dictionary here

while True:
        success, img = cap.read()
        if not success:
            break

        # Process the frame
        results = model(img, stream=True)
        # Define the new resolution - for example, 640x480
        new_width = 290
        new_height = 520

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

                    detected_objects.append(
                        {"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence, "display": True})

        # Merge overlapping objects with class-specific thresholds
        merged_objects = []

        for obj in detected_objects:
            is_merged = False
            # Only process chairs with sufficient confidence
            if obj['class'] == 'chair' and obj['confidence'] < chair_confidence_threshold:
                continue  # Skip this chair detection

            for merged_obj in merged_objects:
                overlap = calculate_overlap(obj['bbox'], merged_obj['bbox'])

                if obj['class'] == 'chair' and merged_obj['class'] == 'chair' and overlap > chair_overlap_threshold:
                    # Merge chairs
                    merged_bbox = merge_boxes(obj['bbox'], merged_obj['bbox'])
                    merged_obj['bbox'] = merged_bbox
                    is_merged = True
                    break

                elif obj['class'] == 'person' and merged_obj['class'] == 'person' and overlap > person_overlap_threshold:
                    # Merge persons
                    merged_bbox = merge_boxes(obj['bbox'], merged_obj['bbox'])
                    merged_obj['bbox'] = merged_bbox
                    is_merged = True
                    break

                elif (obj['class'] in ['backpack', 'bag'] and merged_obj['class'] in ['backpack', 'bag'] and
                      overlap > bag_overlap_threshold):
                    # Merge bags and backpacks
                    merged_bbox = merge_boxes(obj['bbox'], merged_obj['bbox'])
                    merged_obj['bbox'] = merged_bbox
                    is_merged = True
                    break

            if not is_merged:
                merged_objects.append(obj)

        # Check for bag or backpack on chair
        for obj in merged_objects:
            if obj['class'] in ['backpack', 'bag']:
                for chair in merged_objects:
                    if chair['class'] == 'chair':
                        overlap = calculate_overlap(obj['bbox'], chair['bbox'])
                        if overlap > bag_on_chair_overlap_threshold:
                            chair['class'] = 'bag_on_chair'
                            obj['display'] = False
                            break
        # Sort the objects based on the vertical position of their midpoints
        merged_objects = sorted(merged_objects, key=lambda obj: (obj['bbox'][1] + obj['bbox'][3]) / 2, reverse=True)
        merged_objects = sorted(merged_objects, key=lambda obj: (
            (obj['bbox'][1] + obj['bbox'][3]) / 2, (obj['bbox'][0] + obj['bbox'][2]) / 2))

        # Assign 'front' or 'back' labels
        for i, obj in enumerate(merged_objects):
            if i < 3:  # First three are closest to the bottom
                obj['vertical_position'] = 'back'
            else:
                obj['vertical_position'] = 'front'

        # Assign 'left', 'center', or 'right' labels
        sorted_by_horizontal = sorted(merged_objects, key=lambda obj: (obj['bbox'][0] + obj['bbox'][2]) / 2)
        for i, obj in enumerate(sorted_by_horizontal):
            if i < 2:  # First two are closest to the left
                obj['horizontal_position'] = 'left'
            elif i >= len(sorted_by_horizontal) - 2:  # Last two are closest to the right
                obj['horizontal_position'] = 'right'
            else:
                obj['horizontal_position'] = 'center'

        # Initialize counters
        person_count = 0
        chair_count = 0
        bag_on_chair_count = 0

        # Draw bounding boxes
        # Draw bounding boxes
        for obj in merged_objects:
            # Skip drawing individual backpacks, handbags, or chairs if a bag is on a chair
            if 'display' in obj and not obj['display']:
                continue

            bbox = obj['bbox']
            confidence = obj['confidence']

            # Determine the full position label
            position_label = f"{obj.get('vertical_position', '')} {obj.get('horizontal_position', '')}".strip()
            full_label = f"{obj['class']} ({position_label}) - {confidence * 100:.2f}%"

            # Define colors for different classes
            if obj['class'] == 'person':
                color = (0, 255, 0)
                person_count += 1
            elif obj['class'] == 'bag_on_chair':  # Updated label
                color = (255, 255, 0)
                bag_on_chair_count += 1
            elif obj['class'] == 'chair':
                color = (255, 0, 0)
                chair_count += 1
            else:
                color = (255, 0, 0)  # Default color for other classes

            # Draw bounding box, midpoint, and label
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

            # Calculate and draw midpoint
            mid_x = int((bbox[0] + bbox[2]) / 2)
            mid_y = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(img, (mid_x, mid_y), 5, color, -1)

            # Draw the full label
            cv2.putText(img, full_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, full_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Print overlap percentage next to the bounding box
            overlap_text = f"Overlap: {overlap * 100:.2f}%"
            cv2.putText(img, overlap_text, (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # print(f"Persons: {person_count}, Chairs: {chair_count}, Backpacks on Chairs: {backpack_on_chair_count}")

        # Count the number of displayed bounding boxes
        displayed_bbox_count = sum(1 for obj in merged_objects if obj.get('display', True))
        # Define the starting position for the text (top right corner)
        start_x = img.shape[1] - 350  # Start 400 pixels from the right edge; adjust based on your frame size
        start_y = 30  # Start 30 pixels from the top
        line_height = 30  # Space between lines

        # Sort configurations by their count (optional)
        sorted_configs = sorted(config_log.items(), key=lambda x: x[1], reverse=True)[:3]

        # Display each configuration and its count
        for index, (config, count) in enumerate(sorted_configs):
            text = f"Config {index + 1}: {config} - {count} times"
            cv2.putText(img, text, (start_x, start_y + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)



        # Check if exactly 6 bounding boxes are displayed
        if displayed_bbox_count == 6:

            if not screenshot_taken:
                # Save the current frame as a screenshot
                cv2.imwrite("first_six_objects_detected.jpg", img)
                screenshot_taken = True  # Update the flag to avoid taking more screenshots

            position_objects = {'Back': {'Left': 1, 'Center': 1, 'Right': 1},
                                'Front': {'Left': 1, 'Center': 1, 'Right': 1}}

            for obj in merged_objects:
                if obj['class'] == 'chair' and obj.get('display', True):
                    position_key = obj['vertical_position'].capitalize()
                    horizontal_position = obj['horizontal_position'].capitalize()
                    position_objects[position_key][horizontal_position] = 0

            # Print the binary representation of chair presence
            chair_presence = [position_objects[row][pos] for row in ['Back', 'Front'] for pos in
                              ['Left', 'Center', 'Right']]
            print(chair_presence)

            # Log the configuration
            config_str = str(chair_presence)
            if config_str in config_log:
                config_log[config_str] += 1
            else:
                config_log[config_str] = 1

            all_positions_filled = all(
                position is not None for row in position_objects.values() for position in row.values())


            # Inside the loop where new configurations are detected
            if all_positions_filled:
                current_configuration = [position_objects[row][pos] for row in ['Back', 'Front'] for pos in
                                         ['Right', 'Center', 'Left']]

                # Add the current configuration to the list and keep only the last 3
                last_configurations.append(current_configuration)
                last_configurations = last_configurations[-3:]

                # Convert the configuration to a string representation
                config_str = str(current_configuration)

                # Check if this configuration is encountered for the first time
                if config_str not in encountered_configurations:
                    # Log the configuration with an ID
                    config_log[config_str] = config_id_counter

                    # Take a screenshot with bounding boxes
                    ret, frame = cap.read()
                    take_screenshot_with_boxes(frame, merged_objects, f"config_{config_id_counter}.jpg")

                    # Add the configuration to the set of encountered configurations
                    encountered_configurations.add(config_str)

                    # Increment configuration ID counter
                    config_id_counter += 1

        # Display the webcam frame
        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            print(f"Received message: getResults")
            print("Seat Detection requested by MQTT")
            break

cap.release()
cv2.destroyAllWindows()

#publish_top_configuration(config_log)

# After processing all frames, identify the top configuration
max_frequency = 0
top_config = None
for config, frequency in config_log.items():
     if frequency > max_frequency:
        max_frequency = frequency
        top_config = config

print("The following Configurations were measured:")
print(config_log)
print("The following configuration has the highest frequency:")
print(top_config)


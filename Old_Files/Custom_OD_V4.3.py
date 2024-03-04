# Import essential libraries
import sys
import os
import cv2  # For video capture and image processing
import paho.mqtt.client as mqtt  # MQTT protocol for communication with external apps
from ultralytics import YOLO  # YOLO for object detection
import RPi.GPIO as GPIO  # Control Raspberry Pi GPIO pins

# Debugging: Print the Python executable path and version
print(sys.executable)
print(sys.version)

from mqtt_publish import publish_top_configuration  # Import custom function for MQTT publishing

# ------------------------------------------------
# INITIALIZATION SECTION for the Empty Seats Application

# Define GPIO pins for LED indicators to visually represent the application's status
led_pin_start = 18  # Indicates the application has started
led_pin_mqtt = 23   # Indicates a successful MQTT connection
led_pin_webcam = 24 # Indicates the webcam is currently active

# Configuration for connecting to the MQTT broker
broker_address = "broker.hivemq.com"  # Public broker address
broker_port = 1883  # Standard MQTT port
topic_subscribe = "emptySeats/AppToHardware"  # Topic to receive commands from the app
topic_publish = "emptySeats/HardwareToApp"    # Topic to send data to the app

# Control flag for the detection process
start = 0  # Initially set to 0, indicating the detection process is not active

# YOLO model configuration specific to the empty seats detection task
classNames = [list of object classes]  # Predefined classes in YOLO model
desired_classes = ["person", "chair","backpack", "handbag"]  # Target classes for detecting empty seats

# Overlap thresholds for considering objects as overlapping
chair_confidence_threshold = 0.5  # Confidence threshold for detecting chairs
person_confidence_threshold = 0.5  # Confidence threshold for detecting persons
chair_overlap_threshold = 0.5  # Overlap threshold for merging chair detections
person_overlap_threshold = 0.5  # Overlap threshold for merging person detections
bag_overlap_threshold = 0.5  # Overlap threshold for merging bag detections
bag_on_chair_overlap_threshold = 0.5  # Overlap threshold for bag on chair detection



# Storage for configurations (detected objects) over time
last_configurations = []  # List to store recent configurations
config_log = {}  # Dictionary to log configurations for analysis

# ------------------------------------------------
# SETUP SECTION for the Empty Seats Application

# GPIO setup for LEDs
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin-numbering scheme
GPIO.setup(led_pin_start, GPIO.OUT)  # Setup LED pins as output
GPIO.setup(led_pin_mqtt, GPIO.OUT)
GPIO.setup(led_pin_webcam, GPIO.OUT)

# Initial LED state: Start LED on, others off
GPIO.output(led_pin_start, GPIO.HIGH)  # Indicate the application has started
GPIO.output(led_pin_mqtt, GPIO.LOW)  # MQTT connection indicator off
GPIO.output(led_pin_webcam, GPIO.LOW)  # Webcam activity indicator off

# Load the YOLO model with pre-trained weights for object detection
model = YOLO("yolo-Weights/yolov8n.pt")  # Path to YOLO model weights

# Webcam setup for capturing real-time video
video_path = 'seat_test.mp4'  # Placeholder for testing with a video file
cap = cv2.VideoCapture(0)  # Initialize webcam capture (0 for default webcam)

# Verify webcam availability
if not cap.isOpened():
    print("Error: Couldn't open webcam.")
    sys.exit()

# ------------------------------------------------
# MQTT CALLBACK FUNCTIONS for the Empty Seats Application

def on_connect(client, userdata, flags, rc):
    """
    Callback function for when the client connects to the MQTT broker.
    It subscribes to a topic to receive commands and publishes a readiness message.
    """
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe(topic_subscribe)  # Subscribe to receive commands
    client.publish(topic_publish, "PiReady")  # Notify app of readiness
    GPIO.output(led_pin_mqtt, GPIO.HIGH)  # Indicate MQTT connection
def on_message(client, userdata, message):
    """
    Callback function for when a message is received from the MQTT broker.
    It processes commands from the app to start or stop the detection process.
    """
    global start
    message_payload = str(message.payload.decode("utf-8"))
    if message_payload == "Start":
        start = 1  # Start the detection process
        print("Detection process started.")
    elif message_payload == "Stop":
        start = 0  # Stop the detection process
        print("Detection process stopped.")


def start_webcam(last_configurations):
    # Boolean to check if a screenshot has been taken
    # The screenshot is taken when the configuration is detected and is sent to the App
    screenshot_taken = False
    config_hit_20 = False

    while True:
        # The third LED lights up when the webcam is started
        # GPIO.output(led_pin_webcam, GPIO.HIGH)

        # Read the frame from the webcam
        success, img = cap.read()
        if not success:
            break

        # Perform object detection
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

                    detected_objects.append(
                        {"class": class_name, "bbox": (x1, y1, x2, y2), "confidence": confidence, "display": True})

        # Merge overlapping objects with different thresholds for chairs and persons
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
                elif obj['class'] == 'person' and merged_obj[
                    'class'] == 'person' and overlap > person_overlap_threshold:
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

        # Count the number of displayed bounding boxes
        displayed_bbox_count = sum(1 for obj in merged_objects if obj.get('display', True))

        # Check if exactly 6 bounding boxes are displayed
        if displayed_bbox_count == 6:

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

                if config_log[config_str] >= 4:
                    GPIO.output(led_pin_mqtt, GPIO.HIGH)

                if config_log[config_str] >= 80:
                    config_hit_20 = True

            else:
                config_log[config_str] = 1

            all_positions_filled = all(
                position is not None for row in position_objects.values() for position in row.values())

            if all_positions_filled:
                current_configuration = [position_objects[row][pos] for row in ['Back', 'Front'] for pos in
                                         ['Right', 'Center', 'Left']]

                # Add the current configuration to the list and keep only the last 3
                last_configurations.append(current_configuration)
                last_configurations = last_configurations[-3:]

            # Take a screenshot of the current frame and save it in the project's directory
            if screenshot_taken == False:
                script_dir = os.path.dirname(__file__)
                screenshot_path = os.path.join(script_dir, "screenshot.jpg")
                cv2.imwrite(screenshot_path, img)
                screenshot_taken = True
                print('screenshot!')

        # Display the webcam frame
        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) == ord('q'):
            print(f"Received message: getResults")
            print("Seat Detection requested by MQTT")
            break

        if config_hit_20:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Publish the most frequent configuration and a screenshot of it to MQTT
    publish_top_configuration(config_log)

    # Print summary of all configurations for System Evaluation
    print("Summary of Configurations:")
    for config_str, config_id in config_log.items():
        # Convert the configuration string back to a list
        config_array = eval(config_str)
        print(f"Configuration {config_id}: {config_array} - Count: {config_log[config_str]}")


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


def merge_boxes(boxA, boxB):
    return (
        min(boxA[0], boxB[0]),  # x1
        min(boxA[1], boxB[1]),  # y1
        max(boxA[2], boxB[2]),  # x2
        max(boxA[3], boxB[3])  # y2
    )


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


# The main function
def main():
    # Create an MQTT client instance
    client = mqtt.Client()

    # Set the callback functions
    client.on_connect = on_connect
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(broker_address, broker_port, 60)

    # Start the loop
    client.loop_forever()


# ----------
# MQTT client setup
client = mqtt.Client("EmptySeatsDetector")  # Create MQTT client instance
client.on_connect = on_connect  # Assign callback function for connect event
client.on_message = on_message  # Assign callback function for message event
client.connect(broker_address, broker_port, 60)  # Connect to MQTT broker
client.loop_start()  # Start the network loop in a separate thread

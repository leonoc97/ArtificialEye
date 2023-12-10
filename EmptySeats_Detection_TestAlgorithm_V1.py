from ultralytics import YOLO
import cv2
import math


#Either use webcam (1) or video (2). Rest of the code remains the same,
# you just need to comment out A or B

###..A..### Use webcam #####
#cap = cv2.VideoCapture(1)
#cap.set(3, 640)
#cap.set(4, 480)

###..B..#### Use videos #####
video_path = 'video1.mp4'
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

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # filter classes of interest
    chairs = [box for r in results for box in r.boxes if classNames[int(box.cls[0])] == "chair"]
    persons = [box for r in results for box in r.boxes if classNames[int(box.cls[0])] == "person"]

    # coordinates for displaying text
    text_position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    font_thickness = 2

    # check number of detections
    if len(persons) == 3 and len(chairs) == 0:
        cv2.putText(img, "Occupied", text_position, font, font_scale, font_color, font_thickness)
    elif len(chairs) == 1:
        cv2.putText(img, "Free Chair Available", text_position, font, font_scale, font_color, font_thickness)

    # draw bounding boxes for chairs and persons
    for box in chairs + persons:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv2.imshow('Webcam', img)
    cv2.putText(img, "Free Chair Available", text_position, font, font_scale, font_color, font_thickness)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import pyttsx3
from playsound import playsound

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Access camera
cap = cv2.VideoCapture(0)

detected_objects = []  # Array to store detected objects

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# User input for the object to detect
object_to_detect = input("Enter the object you want to detect: ").strip().lower()

# Camera calibration parameters (you need to determine these values)
focal_length = 1000  # Focal length in pixels
known_object_height = 0.2  # Height of the known object in meters
known_distance = 1.0  # Known distance to the object in meters

# Function to calculate distance
def calculate_distance(object_height_pixels):
    distance = (known_object_height * focal_length) / object_height_pixels
    return distance

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare input for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass through the network
    outputs = net.forward(output_layer_names)

    # Initialize lists to store detection info
    boxes = []
    confidences = []
    class_ids = []

    # Process YOLO outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Minimum confidence threshold
                label = str(classes[class_id])

                if label == object_to_detect:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Notify and draw bounding box if the specified object is detected
    if object_to_detect and len(boxes) > 0:
        playsound("notification.mp3")  # Replace "notification.mp3" with the path to your notification sound
        object_info = f"{object_to_detect.capitalize()} detected!"
        engine.say(object_info)
        engine.runAndWait()

        # Calculate and display the distance of the detected object
        distance = calculate_distance(boxes[0][3])  # Assuming only one object is detected
        distance_info = f"Distance: {distance:.2f} meters"
        engine.say(distance_info)
        engine.runAndWait()

        # Draw bounding box, label, and distance on the frame
        x, y, w, h = boxes[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{object_to_detect.capitalize()} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Distance: {distance:.2f} meters', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Object Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

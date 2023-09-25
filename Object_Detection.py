import cv2
import numpy as np
import pyttsx3

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

    # Non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Append detected object information to the array
        detected_objects.append({
            "label": label,
            "confidence": confidence,
            "height_pixels": h
        })

        # Calculate distance based on detected object's height
        distance = calculate_distance(h)

        # Speak the detected object information with distance
        object_info = f"Label: {label}, Confidence: {confidence:.2f}, Height: {h} pixels, Distance: {distance:.2f} meters"
        engine.say(object_info)
        engine.runAndWait()

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Height: {h} pixels, Distance: {distance:.2f} meters', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Object Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Display the array of detected objects
for obj in detected_objects:
    print(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}, Height: {obj['height_pixels']} pixels, Distance: {calculate_distance(obj['height_pixels']):.2f} meters")

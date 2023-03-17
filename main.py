import cv2

# Load the video file and create a foreground detector
video = cv2.VideoCapture('traffic.avi')
object_detector = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=1000)

# Define minimum size threshold for the detected objects
min_size = 1000

def detect_cars(frame):
    # Detect the foreground objects
    object_mask = object_detector.apply(frame)

    # Remove noise from the detected object
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    noise_free_object = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, structure)

    # Analyze the blobs in the image and draw bounding boxes around the cars
    contours, hierarchy = cv2.findContours(noise_free_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_size]

    # Draw bounding boxes around the cars
    detected_frame = frame.copy()
    for box in bounding_boxes:
        detected_frame = cv2.rectangle(detected_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

    # Count the number of cars and display the result
    number_of_cars = len(bounding_boxes)
    detected_frame = cv2.putText(detected_frame, f"Number of Cars: {number_of_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return detected_frame

# Process the video and display the detected cars
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    detected_frame = detect_cars(frame)

    cv2.imshow('Detected Cars', detected_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
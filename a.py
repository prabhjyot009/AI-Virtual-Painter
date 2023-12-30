import cv2
import numpy as np
from tensorflow.keras.models import load_model
import HandTrackingModule as htm
import os
import time

# Load the trained CNN model
model = load_model('digit_recognition_model.h5')

# Brush and eraser thickness
brush_thickness = 15
eraser_thickness = 50

# Previous and current time for FPS calculation
p_time = 0
c_time = 0

# Folder path containing header images
folder_path = "Header"
my_list = os.listdir(folder_path)
overlay_list = [cv2.imread(f'{folder_path}/{im_path}') for im_path in my_list]

# Default header image and draw color
header = overlay_list[0]
draw_color = (0, 0, 255)

# Video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = htm.handDetector(detectionCon=0.75)

# Canvas for drawing
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Get the bounding box coordinates of the index finger
    x, y, w, h = detector.get_finger_coordinates(img)

    # Check if the bounding box is valid
    if w > 0 and h > 0:
        # Extract the region where the user draws the digit
        digit_roi = img[y:y + h, x:x + w]

        # Check if digit_roi is not empty
        if digit_roi.size != 0:
            # Preprocess the drawn digit image
            processed_digit = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
            processed_digit = cv2.resize(processed_digit, (28, 28))
            processed_digit = processed_digit.astype('float32') / 255.0
            processed_digit = np.expand_dims(processed_digit, axis=0)

            # Use the trained CNN model to predict the digit
            prediction = model.predict(processed_digit)
            predicted_digit = np.argmax(prediction)

            # Display the predicted digit on the screen
            cv2.putText(img, f'Prediction: {predicted_digit}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Blend canvas with the video feed
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Calculate and display FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (255, 0, 255), 3)

    # Set the header image
    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    cv2.waitKey(1)

'''
Virtual Mouse Test Script
This script implements the virtual mouse using the custom model created in 
`virtual_mouse.ipynb`. The mouse implements cursor movement, left-clicks,
and right-clicks.
NOTE: Ensure the `.task` model is in the same folder as this script before 
running. Custom model is `gesture_recognizer.task`; found in the Google 
Drive project folder.
NOTE: Probably begin with the hand in the point pose. Model was trained on
both the left and right hand so the virtual mouse should theoretically
be able to handle either hand (not both).
Authors: Emanuel Charro, JoseAngel Pulido
Last Updated: 03/06/2026
'''

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time

# 1. CONFIGURATION =========================================================
# Disable PyAutoGUI failsafe so script doesn't crash if pointing to a corner
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# Camera settings
cam_w, cam_h = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

# Point to the model
# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join that directory with the model filename
model_path = os.path.join(script_dir, 'gesture_recognizer.task')

# Smoothing variables to prevent cursor jitter
prev_mouse_x, prev_mouse_y = 0, 0
curr_mouse_x, curr_mouse_y = 0, 0
smoothing_factor = 5

# State trackers to prevent "spam clicking" while holding the pose
left_click_held = False
right_click_held = False

# FPS tracking variable
pTime = 0

# 2. INITIALIZE CUSTOM GESTURE RECOGNIZER ==================================
base_options = python.BaseOptions(model_asset_path=model_path)
# Running in IMAGE mode allows us to process frames synchronously in main loop
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
recognizer = vision.GestureRecognizer.create_from_options(options)

print("Virtual Mouse Active. Press 'q' in the window to exit.")

# 3. MAIN LOOP =============================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process the image with your custom AI model
    result = recognizer.recognize(mp_image)

    if result.hand_landmarks and result.gestures:
        # Extract the highest-confidence gesture
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        confidence = top_gesture.score

        # Extract landmarks to move the cursor
        landmarks = result.hand_landmarks[0]
        index_tip = landmarks[8] # Index finger tip

        # 1. MOVE THE MOUSE (Always track the index finger) ----------------
        # Convert landmark coordinates (0.0 to 1.0) to screen resolution
        target_x = np.interp(index_tip.x, [0, 1], [0, screen_w])
        target_y = np.interp(index_tip.y, [0, 1], [0, screen_h])

        # Apply a smoothing filter
        curr_mouse_x = prev_mouse_x + (target_x - prev_mouse_x) / smoothing_factor
        curr_mouse_y = prev_mouse_y + (target_y - prev_mouse_y) / smoothing_factor
        
        pyautogui.moveTo(curr_mouse_x, curr_mouse_y)
        prev_mouse_x, prev_mouse_y = curr_mouse_x, curr_mouse_y

        # 2. HANDLE CLICKS BASED ON AI CLASSIFICATION ----------------------
        # Model Maker uses the same names of the dataset folders as the labels
        
        if gesture_name == 'left-click':
            cv2.putText(frame, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if not left_click_held:
                pyautogui.click(button='left')
                left_click_held = True
        else:
            left_click_held = False # Reset when the gesture stops

        if gesture_name == 'right-click':
            cv2.putText(frame, "RIGHT CLICK", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not right_click_held:
                pyautogui.click(button='right')
                right_click_held = True
        else:
            right_click_held = False

        if gesture_name == 'point':
            cv2.putText(frame, "POINT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        if gesture_name == 'none':
            cv2.putText(frame, "NONE (Ignored)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)

        # Draw a circle on the index tip to see what is driving the cursor
        px_x = int(index_tip.x * cam_w)
        px_y = int(index_tip.y * cam_h)
        cv2.circle(frame, (px_x, px_y), 10, (255, 0, 0), cv2.FILLED)

    # Calculate and Display FPS --------------------------------------------
    cTime = time.time()
    # Prevent division by zero if the loop processes instantaneously
    if cTime != pTime:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0
    pTime = cTime
    
    # Place FPS counter in the top-right corner
    cv2.putText(frame, f'FPS: {int(fps)}', (cam_w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Custom Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
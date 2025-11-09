"""
Project Name: Drowsiness Detection System (EYES ON WHEELS)
Team Members:
    - JAY CHHANTBAR (22SE02CS006)
    - SAMARTH DUMASIA (22SE02CS011)
    -JENEEL DUMASIA (22SE02CS012)
    -JAYKUMAR MISTRY (22SE02CS028)
    -DARSHILKUMAR PATEL (22SE02CS031)
University/Institution: P. P. SAVANI UNIVERSITY
Department: School of Engineering
Course: B.Tech (Computer Science and Engineering)

Project Supervisor: [ABHIJITSINH PARMAR]

Description: This project aims to implement a real-time drowsiness detection system using IoT and cameras to monitor driver attentiveness.
"""

import numpy as np
import cv2
import pygame
import dlib
from scipy.spatial import distance as dist

# Initialize Pygame for sound
pygame.init()
pygame.mixer.init()

# Load the pre-trained dlib face detector and shape predictor for landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Directly use GStreamer pipeline for accessing the camera
gst_pipeline = (
    "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera")
    exit()

# Load alarm sound
alarm = pygame.mixer.Sound('alarm.wav')
alarm_playing = False
count = 0

# Eye aspect ratio threshold to detect drowsiness
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 10

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Get the indexes for the eyes landmarks
(lStart, lEnd) = (42, 48)  # Left eye
(rStart, rEnd) = (36, 42)  # Right eye

# Function for applying CLAHE
def apply_clahe(gray_frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale and apply CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)  # Normalize lighting
    gray = cv2.bilateralFilter(gray, 5, 1, 1)  # Reduce noise

    # Detect faces
    rects = detector(gray, 0)

    if len(rects) > 0:
        for rect in rects:
            # Detect facial landmarks
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # Extract left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Compute the eye aspect ratio for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio
            ear = (leftEAR + rightEAR) / 2.0

            # Visualize the eyes on the frame
            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 2)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 2)

            # Check if EAR is below the threshold, indicating drowsiness
            if ear < EYE_AR_THRESH:
                count += 1
                cv2.putText(frame, "Drowsy!", (100, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                if count >= EYE_AR_CONSEC_FRAMES and not alarm_playing:
                    alarm.play()
                    alarm_playing = True
            else:
                count = 0
                cv2.putText(frame, "Eyes open!", (100, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
                if alarm_playing:
                    alarm.stop()
                    alarm_playing = False

    else:
        cv2.putText(frame, "No face detected", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
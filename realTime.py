# Importing OpenCV for computer vision stuff
import cv2

# Importing matplotlib for visualizing captured images
from matplotlib import pyplot as plt

# Importing warnings to ignore it XD
import warnings
warnings.filterwarnings('ignore')

# Import the YOLO module from the ultralytics library.
from ultralytics import YOLO

import os
import torch
import time

# Image indexes will be:
img_indx_low = 1
img_indx_high = 6

# Creating a function for all this work
file_path = 'images/'

# Confidence number
confidence = 3

# .pt is pytorch extension for saving the weights and other values.
# 'n' stands for nano. There are many architectures for YOLO like 's' for small, 'l' (large), 'x' (extra large), 'm' (medium)
model = YOLO('yolov8n.pt') 

# Defining capture object and its setting
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Number of attempts inside while loop
attempt = 0

def img_exists(indx_low, indx_high, file_path):
    for i in range(indx_low,indx_high):
        file_path += f"image{i}.jpg"
        if os.path.exists(file_path):
            os.remove(file_path)
        file_path = file_path[:7]

def take_photo(): 
    img_exists(img_indx_low, img_indx_high, file_path)   
    is_running = False
    for i in range(img_indx_low,img_indx_high):

        # If frames aren't read fast enough, the camera buffer fills up with old frames
        # causing a delay between real-time actions and captured images. To avoid this, calling cap.read() without storing its results
        for _ in range(2):
            cap.read()
        is_frame, frame = cap.read()
        if (is_frame):
            is_running = True
            cv2.imwrite(f"images/image{i}.jpg", frame)

    return is_running

# Returning results for each image
def is_human_count():
    count = 0
    for i in range(img_indx_low, img_indx_high):
        results = model(f'images/image{i}.jpg', verbose = False)  # Verbose refers to how much detail a program or function prints to the screen.
    
        # checking if the result is human (i.e., index 0)
        boxes = results[0].boxes
        class_ids = boxes.cls  # this is a tensor of class IDs

        if torch.any(class_ids == 0):
            count += 1

    return count

while(True):
    # Calling the function
    if(not take_photo()):
        print("Webcam not found!")
        break
    else:
        if(is_human_count() >= confidence):
            print(f"Human Detected!") # Which will be the signal to Rasberry Pi

        else:
            print(f"No Human Detected!")

    attempt += 1
    time.sleep(3)

cap.release()
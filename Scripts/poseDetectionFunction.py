import math
import os
import cv2
import numpy as np
from time import time
import mediapipe as mp  
import matplotlib.pyplot as plt
from pathlib import Path


mp_pose = mp.solutions.pose #Clase

#Pose Function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, display=True):
    #Create a copy of the input image
    output_image = image.copy()
    
    #Convert the image from BGR into RGB format
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Process the RGB image to detect the pose
    results = pose.process(imgRGB)
    
    height, width, _ = image.shape 
    landmarks = []

    #Check if any landmarks are detected
    if results.pose_landmarks:
        #Draw Pose landmarks on the output image
        mp_drawing.draw_landmarks(image = output_image,landmark_list = results.pose_landmarks, connections =  mp_pose.POSE_CONNECTIONS)
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))

    #Display the output image
    if display:
        plt.figure(figsize=[10,10])
        plt.subplot(121);plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off')
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]); plt.title("Original Image"); plt.axis('off')
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks
    

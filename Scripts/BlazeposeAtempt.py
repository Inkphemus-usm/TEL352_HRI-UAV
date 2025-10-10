import math
import cv2
import numpy as np
from time import time
import mediapipe as mp  
import matplotlib.pyplot as plt



mp_pose = mp.solutions.pose #Clase

#Pose Function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

sample_image = cv2.imread('sample.jpg')

plt.figure(figsize=[10,10])

plt.title("Sample Image"); plt.axis('off'); plt.imshow(sample_image[:,:,::-1]); plt.show()

results = pose.process(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))

img_copy = sample_image.copy()

if results.pose_landmarks:
    mp_drawing.draw_landmarks(image = img_copy,landmark_list = results.pose_landmarks, connections =  mp_pose.POSE_CONNECTIONS)
    plt.figure(figsize=[10,10])
    plt.title("output"); plt.axis('off'); plt.imshow(img_copy[:,:,::-1]); plt.show()
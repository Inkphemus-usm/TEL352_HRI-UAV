import math
import os
import cv2
import numpy as np
from time import time
import mediapipe as mp  
import matplotlib.pyplot as plt
from pathlib import Path
from poseDetectionFunction import detectPose

mp_pose = mp.solutions.pose #Clase

#Pose Function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils


sample_path = Path(__file__).resolve().parent.parent.joinpath('media', 'sample.jpg')
sample_path = sample_path.resolve()
if not sample_path.exists():
	print(f"Sample image not found at {sample_path}")
	raise SystemExit(1)

image = cv2.imread(str(sample_path))
if image is None:
	print(f"Failed to read image (corrupt or unsupported): {sample_path}")
	raise SystemExit(1)

detectPose(image, pose, display=True)
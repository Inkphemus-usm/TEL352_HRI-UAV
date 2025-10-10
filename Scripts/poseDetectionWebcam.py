import math
import os
import cv2
import numpy as np
from time import time
import mediapipe as mp  
import matplotlib.pyplot as plt
from poseDetectionFunction import detectPose



mp_pose = mp.solutions.pose #Clase

#Pose Function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils
   
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Try to open a camera. Allow overriding with env var CAMERA_INDEX. If no camera
# opens, try indices 0..5 and report a clear error.
cam_index_env = os.environ.get('CAMERA_INDEX')
def open_camera(preferred=None, max_try=5):
    if preferred is not None:
        cap = cv2.VideoCapture(int(preferred))
        if cap.isOpened():
            return cap, int(preferred)
        cap.release()
    for i in range(0, max_try+1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
        cap.release()
    return None, None

video, opened_index = open_camera(cam_index_env, max_try=5)
if video is None:
    raise RuntimeError('Could not open any camera. Tried CAMERA_INDEX (env) and indices 0..5.')

cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
video.set(3,1280)
video.set(4,960)

time1 = 0.0
while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break
    
    # Detect Pose
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    # Preserve aspect ratio, target height 640
    new_h = 640
    new_w = int(frame_width * (new_h / frame_height))
    frame = cv2.resize(frame, (new_w, new_h))
    frame, _ = detectPose(frame, pose_video, display=False)

    time2 = time()
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
        cv2.putText(frame, 'FPS: {}'.format(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    time1 = time2
    cv2.imshow('Pose Detection', frame)
    k = cv2.waitKey(1) & 0xFF

    if (k == 27):
        break
video.release()
cv2.destroyAllWindows()
import math
import os
import cv2
import numpy as np
from time import time
import mediapipe as mp  
import matplotlib
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

    # Define which landmark indices to keep: arms and torso
    # MediaPipe Pose indices: shoulders(11,12), elbows(13,14), wrists(15,16), hips(23,24)
    selected_indices = [11,12,13,14,15,16,17,18,19,20,23,24]

    #Check if any landmarks are detected
    if results.pose_landmarks:
        # Build a list of selected landmarks (pixel coordinates)
        lm_list = results.pose_landmarks.landmark
        for idx in selected_indices:
            if idx < len(lm_list):
                lm = lm_list[idx]
                x_px = int(lm.x * width)
                y_px = int(lm.y * height)
                z_val = lm.z * width
                landmarks.append((x_px, y_px, z_val))

        # Draw simplified skeleton (only arms and torso)
        # Left arm: 11->13->15 ; Right arm: 12->14->16
        # Torso connections: 11->12 (shoulders), 11->23 (left shoulder to left hip), 12->24 (right shoulder to right hip), 23->24 (hips)
        def safe_point(i):
            # return tuple or None
            try:
                return landmarks[selected_indices.index(i)]
            except ValueError:
                return None

        # draw circles for each selected landmark
        for (x_px, y_px, _) in landmarks:
            cv2.circle(output_image, (x_px, y_px), 5, (0, 255, 0), -1)

        # helper to draw line between two landmark indices if both present
        def draw_conn(a, b, color=(0,200,0), thickness=2):
            pa = safe_point(a)
            pb = safe_point(b)
            if pa is not None and pb is not None:
                cv2.line(output_image, (pa[0], pa[1]), (pb[0], pb[1]), color, thickness)

        # arms
        draw_conn(11,13)
        draw_conn(13,15)
        draw_conn(12,14)
        draw_conn(14,16)
        # torso
        draw_conn(11,12, color=(255,0,0), thickness=3)
        draw_conn(11,23, color=(255,0,0), thickness=3)
        draw_conn(12,24, color=(255,0,0), thickness=3)
        draw_conn(23,24, color=(255,0,0), thickness=3)

    #Display the output image (show only original and simplified output)
    if display:
        plt.figure(figsize=[10,10])
        plt.subplot(121);plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off')
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]); plt.title("Simplified Landmarks"); plt.axis('off')
        # If an X/Wayland display is available, try to show interactively; otherwise save to media/
        out_path = Path(__file__).resolve().parent.joinpath('..', 'media', 'output_sample.png').resolve()
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            backend = matplotlib.get_backend()
            # If backend is non-interactive (Agg) we cannot show; save instead
            if 'agg' in backend.lower():
                # attempt to show may not work; save to file instead
                cv2.imwrite(str(out_path), output_image)
                print(f"Non-interactive matplotlib backend ('{backend}'). Wrote output image to: {out_path}")
            else:
                try:
                    plt.show()
                except Exception:
                    # fallback to saving
                    cv2.imwrite(str(out_path), output_image)
                    print(f"plt.show() failed. Wrote output image to: {out_path}")
        else:
            # Save the output image so the user can inspect it when running headless
            cv2.imwrite(str(out_path), output_image)
            print(f"No graphical display detected. Wrote output image with simplified landmarks to: {out_path}")
    else:
        return output_image, landmarks
    

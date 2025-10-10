

import math
import os
import sys
import logging
import traceback
# Ensure Qt uses an offscreen platform (fixes "Could not load the Qt platform plugin 'xcb'" in headless/WSL)
# Must be set before importing cv2 so OpenCV's Qt plugin doesn't try to initialize a GUI platform.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
import numpy as np
from time import time, sleep
import mediapipe as mp  
import matplotlib.pyplot as plt
from poseDetectionFunction import detectPose



mp_pose = mp.solutions.pose #Clase

#Pose Function
#pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

pose_video= mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
# Log that the Mediapipe pose model has been created
print('Pose model initialized')

# Configure logging so we can see progress and errors in the console (and optionally a file)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def try_open_camera(indices=(0,), backends=(cv2.CAP_V4L2, cv2.CAP_ANY)):
    """Try opening a camera using different backends and indices.
    Returns (cap, used_index, used_backend) or (None, None, None).
    """
    for backend in backends:
        for idx in indices:
            try:
                cap = cv2.VideoCapture(int(idx), backend)
            except Exception:
                # some OpenCV builds expect a different signature
                cap = cv2.VideoCapture(int(idx))
            if cap is not None and cap.isOpened():
                return cap, idx, backend
            if cap is not None:
                cap.release()
    return None, None, None

# Try to open the camera. On WSL use V4L2 explicitly first, then fallback.
video, used_index, used_backend = try_open_camera(indices=(0,1,2), backends=(cv2.CAP_V4L2, cv2.CAP_ANY))
if video is None or not video.isOpened():
    logger.warning('Camera device could not be opened with tried backends/indices. Trying sample file fallback...')
    # fallback to a sample file if camera is not available
    video = cv2.VideoCapture('videoSample.mp4')
    if video is None or not video.isOpened():
        logger.error('Fallback videoSample.mp4 could not be opened. Aborting.')
    else:
        logger.info('Opened videoSample.mp4 as fallback source')
else:
    logger.info('VideoCapture opened successfully (index=%s, backend=%s)', used_index, used_backend)

# If camera opened but reads timeout (common on WSL), try a short warm-up read loop
def warmup_capture(cap, timeout_s=8.0, poll_interval=0.2):
    start = time()
    attempts = 0
    while time() - start < timeout_s:
        ok, frame = cap.read()
        attempts += 1
        if ok and frame is not None:
            logger.info('Warm-up read succeeded after %d attempts', attempts)
            return True, frame
        sleep(poll_interval)
    logger.warning('Warm-up read failed after %d attempts (%.1fs)', attempts, timeout_s)
    return False, None

if video is not None and video.isOpened():
    ok_warm, _ = warmup_capture(video, timeout_s=8.0, poll_interval=0.25)
    if not ok_warm:
        # If warmup fails and we used a camera device, try fallback to sample file
        if used_index is not None:
            logger.warning('Camera opened but no frames received. Switching to sample video for debugging.')
            video.release()
            video = cv2.VideoCapture('videoSample.mp4')
            if video is None or not video.isOpened():
                logger.error('Could not open fallback sample video either.')
            else:
                logger.info('Opened videoSample.mp4 as fallback source')

# If Qt is forced to 'offscreen', don't call cv2.imshow (it will fail). Instead
# write processed frames to an output video file so the script can run headless.
is_offscreen = os.environ.get('QT_QPA_PLATFORM', '') == 'offscreen'
writer = None

# If running headless we'll write an output file so you can inspect results later
output_path = 'output_pose.mp4'

frame_count = 0
time1 = 0
try:
    logger.info('Starting video processing (offscreen=%s)', is_offscreen)
    # Try to get source FPS as a fallback
    source_fps = video.get(cv2.CAP_PROP_FPS) or 0
    if source_fps <= 0:
        source_fps = 25.0
    logger.info('Source FPS (fallback applied if 0): %s', source_fps)
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            logger.info('End of video or cannot read frame (ok=%s).', ok)
            break

        frame_count += 1
        # Preprocess frame
        frame = cv2.flip(frame, 1)
        fh, fw, _ = frame.shape
        logger.debug('Original frame size: %dx%d -> resizing to height 640', fw, fh)
        frame = cv2.resize(frame, (int(fw * (640/fh)), 640))

        try:
            logger.debug('About to call detectPose() for frame %d', frame_count)
            processed_frame, landmarks = detectPose(frame, pose_video, display=False)
            logger.debug('detectPose() returned for frame %d (landmarks=%d)', frame_count, len(landmarks))
        except Exception as e:
            # Catch errors coming from mediapipe or drawing
            logger.error('Error during pose detection on frame %d: %s', frame_count, e)
            logger.debug(traceback.format_exc())
            break

        time2 = time()
        fps = 0
        if (time2 - time1) > 0:
            fps = 1.0 / (time2 - time1)
            cv2.putText(processed_frame, 'FPS: {}'.format(int(fps)), (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
        time1 = time2

        # Initialize writer on first frame when running headless
        if is_offscreen:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Use detected FPS when available otherwise source_fps
                out_fps = int(fps) if fps > 0 else int(source_fps)
                h, w = processed_frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
                logger.info('Initialized writer -> %s (fps=%s, size=%sx%s)', output_path, out_fps, w, h)
            else:
                logger.debug('Writer already initialized')
            writer.write(processed_frame)
        else:
            # When not headless show the frame so you can see it's running
            cv2.imshow('Pose Detection', processed_frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                logger.info('User requested exit (ESC)')
                break

        # Log a short status every N frames so you can confirm progress
        if frame_count % 30 == 0:
            logger.info('Frame %d processed (fps=%.1f, landmarks=%d)', frame_count, fps, len(landmarks))

except Exception as e:
    logger.exception('Unhandled exception while processing video: %s', e)
finally:
    video.release()
    if writer is not None:
        writer.release()
        logger.info('Wrote output video to %s', output_path)
    cv2.destroyAllWindows()
    logger.info('Processing finished. Total frames: %d', frame_count)
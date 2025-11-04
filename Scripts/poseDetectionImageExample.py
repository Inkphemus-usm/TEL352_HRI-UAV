import math
import os
import sys
import cv2
import numpy as np
from time import time
import mediapipe as mp  
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from poseDetectionFunction import detectPose
import logging

# Configure logging for console feedback
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s %(levelname)s: %(message)s',
	handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose #Clase

#Pose Function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

# Warn if user set USE_OPENVINO but this example uses MediaPipe
if os.environ.get('USE_OPENVINO', '0') == '1':
	logger.warning('Environment variable USE_OPENVINO=1 is set, but this example uses MediaPipe. To run with OpenVINO use Scripts/poseDetectionVideo.py or adapt this script.')

sample_path = Path(__file__).resolve().parent.parent.joinpath('media', 'sample3.jpg')
sample_path = sample_path.resolve()
if not sample_path.exists():
	logger.error('Sample image not found at %s', sample_path)
	logger.info('Place a sample image at media/sample3.jpg or change the sample_path variable in this script.')

image = cv2.imread(str(sample_path))
if image is None:
	logger.error('Failed to read image (corrupt or unsupported): %s', sample_path)
	logger.info('Try opening the image with an image viewer to confirm it is valid.')
	raise SystemExit(1)

logger.info('Loaded sample image: %s (shape=%s)', sample_path, image.shape)
try:
	detectPose(image, pose, display=True)
	logger.info('detectPose completed successfully (display=True).')
except Exception as e:
	logger.exception('Error while running detectPose: %s', e)
	logger.info('If you are running headless, the script will save the output image to media/output_sample.png. Check permissions and paths.')
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


sample_path = Path(__file__).resolve().parent.parent.joinpath('media', 'sample3.jpg')
sample_path = sample_path.resolve()
if not sample_path.exists():
	print(f"Sample image not found at {sample_path}")
	raise SystemExit(1)

image = cv2.imread(str(sample_path))
if image is None:
	print(f"Failed to read image (corrupt or unsupported): {sample_path}")
	raise SystemExit(1)

detectPose(image, pose, display=True)
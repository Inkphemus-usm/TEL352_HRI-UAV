"""Lightweight OpenVINO wrapper to run a pose landmarks model.

This is a best-effort generic wrapper. To use it:
- Provide an OpenVINO IR (.xml + .bin) or an ONNX model path in OPENVINO_MODEL_PATH.
- Set environment variable USE_OPENVINO=1 before running the video script.

The wrapper assumes the model takes an image input and outputs a flat vector of
landmarks (x,y,z) for each keypoint (for example 33*3 values). If your model has a
different I/O, adapt the preprocessing/postprocessing below.
"""
from typing import Tuple, List
import numpy as np
import logging
try:
    import openvino.runtime as ov
except Exception:
    ov = None

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


class OpenVINOPose:
    def __init__(self, model_path: str, input_size=(256, 256), device: str = 'CPU'):
        if ov is None:
            raise RuntimeError('openvino.runtime not available (install openvino).')
        self.core = ov.Core()
        # Read model (ONNX or IR)
        try:
            self.model = self.core.read_model(model=model_path)
        except Exception as e:
            # some OpenVINO versions accept path as str differently
            raise
        # compile for device
        self.compiled = self.core.compile_model(self.model, device)
        # pick input and output names
        self.input = list(self.compiled.inputs)[0]
        self.output = list(self.compiled.outputs)[0]
        self.input_shape = tuple(self.input.shape)  # e.g. (1,3,256,256) or (1,256,256,3)
        self.input_size = input_size
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        logger.info('OpenVINO model loaded: %s -> input_shape=%s output=%s', model_path, self.input_shape, self.output.shape)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # Resize to configured input size
        h, w = self.input_size
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Determine layout expected by model
        if len(self.input_shape) == 4 and self.input_shape[1] == 3:
            # NCHW
            img = np.transpose(img, (2, 0, 1))  # HWC->CHW
            img = np.expand_dims(img, 0)
        else:
            # NHWC or flat
            img = np.expand_dims(img, 0)
        return img

    def _postprocess(self, output: np.ndarray, frame_w: int, frame_h: int) -> List[Tuple[int, int, float]]:
        # Expect output shape (1, N) or (1, K, 3) or similar
        out = np.array(output)
        if out.ndim == 2 and out.shape[0] == 1:
            out = out[0]
        # If can be reshaped to (num_kp,3)
        if out.size % 3 == 0:
            kp = out.reshape(-1, 3)
            landmarks = []
            for x, y, z in kp:
                # If model outputs normalized coords [0,1]
                lx = int(x * frame_w) if x <= 1.0 else int(x)
                ly = int(y * frame_h) if y <= 1.0 else int(y)
                landmarks.append((lx, ly, z))
            return landmarks
        # Fallback: return empty
        return []

    def infer_and_draw(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        # Run inference and draw landmarks on a copy of the frame
        frame_h, frame_w = frame.shape[:2]
        inp = self._preprocess(frame)
        # run
        try:
            # compiled_model([input_array]) -> returns ov.Values or list
            results = self.compiled([inp])
            # results can be a list or dict-like; get first tensor
            if isinstance(results, list) or isinstance(results, tuple):
                out = results[0]
            else:
                # try mapping
                out = next(iter(results.values()))
        except Exception as e:
            logger.exception('OpenVINO inference failed: %s', e)
            return frame, []

        landmarks = self._postprocess(out, frame_w, frame_h)

        # Draw landmarks using mediapipe drawing utilities (best-effort)
        out_img = frame.copy()
        try:
            # Convert landmarks to mediapipe LandmarkList-like for drawing
            from mediapipe.framework.formats import landmark_pb2
            lm = landmark_pb2.NormalizedLandmarkList()
            for x, y, z in landmarks:
                # normalized coordinates expected (0..1)
                nl = lm.landmark.add()
                nl.x = x / frame_w
                nl.y = y / frame_h
                nl.z = z
            self.mp_drawing.draw_landmarks(out_img, lm, self.mp_pose.POSE_CONNECTIONS)
        except Exception:
            # If conversion fails, draw simple circles
            for x, y, z in landmarks:
                cv2.circle(out_img, (int(x), int(y)), 3, (0, 255, 0), -1)

        return out_img, landmarks

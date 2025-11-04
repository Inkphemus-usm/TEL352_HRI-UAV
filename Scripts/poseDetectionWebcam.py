#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose GUI robusta con MediaPipe + OpenCV (Linux/WSL).
- Abre cámara con backend V4L2 y combos de FOURCC/resolución/FPS.
- Dibuja landmarks y conexiones.
- GUI con FPS y atajos: ESC/q = salir, s = guardar frame, f = fullscreen.
- Fallback a archivo de video si no hay cámara disponible.

Autor: tú :)
"""

import os
import sys
import time
import glob
import os.path as osp
import cv2
import numpy as np
import logging

# (opcional) silenciar logs ruidosos de TFLite/absl
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp

# ------------- Configuración -------------
FALLBACK_VIDEO = "media/running.mp4"   # cámbialo si quieres otro respaldo
WINDOW_NAME    = "Pose Detection (ESC/q salir, s guardar, f fullscreen)"
TARGET_HEIGHT  = 720                   # alto de visualización (mantiene aspecto)
PREFERRED_IDX  = 0                     # índice preferido de cámara
MAX_MISSES     = 5                     # lecturas fallidas consecutivas antes de reintentar/cerrar
# -----------------------------------------

mp_pose   = mp.solutions.pose
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Configure logging for console feedback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Warn if USE_OPENVINO is set: this script uses MediaPipe only by default
if os.environ.get('USE_OPENVINO', '0') == '1':
    logger.warning('USE_OPENVINO=1 is set but %s uses MediaPipe only. To use OpenVINO, run Scripts/poseDetectionVideo.py or adapt this script.', __file__)

# Optional OpenVINO support: attempt to initialize wrapper if requested
use_openvino = os.environ.get('USE_OPENVINO', '0') == '1'
openvino_model_path = os.environ.get('OPENVINO_MODEL_PATH', '').strip()
openvino_wrapper = None
if use_openvino and openvino_model_path:
    try:
        from openvino_inference import OpenVINOPose
        openvino_wrapper = OpenVINOPose(openvino_model_path)
        logger.info('OpenVINO wrapper initialized (model=%s)', openvino_model_path)
    except Exception as e:
        logger.warning('OpenVINO initialization failed: %s. Falling back to MediaPipe.', e)
        openvino_wrapper = None
        use_openvino = False


def list_video_indices_linux():
    """Devuelve índices de /dev/video* presentes (Linux/WSL)."""
    devs = sorted(glob.glob("/dev/video*"))
    idxs = []
    for d in devs:
        base = osp.basename(d)
        if base.startswith("video"):
            try:
                idxs.append(int(base.replace("video", "")))
            except Exception:
                pass
    return idxs


def warmup(cap, tries=2, delay=0.15):
    """Lee algunos frames para 'calentar' la cámara."""
    ok = False
    for _ in range(tries):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(delay)
    return ok


def fourcc_to_str(v: float) -> str:
    """Convierte el valor numérico de CAP_PROP_FOURCC a string legible."""
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


def try_configure(cap: cv2.VideoCapture, combos):
    """
    Intenta configurar la cámara con varios combos de (FOURCC, W, H, FPS).
    Devuelve True si alguno funcionó.
    """
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for fourcc, w, h, fps in combos:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)

        if warmup(cap):
            # Lee props reales aplicadas
            real_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            real_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            real_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            real_fc  = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
            print(f"[INFO] Cámara OK con {real_fc} {real_w}x{real_h}@{real_fps:.1f}")
            return True
    return False


def open_camera_with_v4l2(idx: int):
    """
    Abre una cámara con backend V4L2 y configura combos comunes.
    Devuelve el cap abierto o None.
    """
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None

    # Combos típicos compatibles (ajusta según tu 'v4l2-ctl --list-formats-ext')
    combos = [
        ("MJPG", 1280, 720, 30),
        ("MJPG",  640, 480, 30),
        ("YUYV",  640, 480, 30),  # YUY2/YUYV
        ("MJPG", 1920,1080, 30),
        ("MJPG",  800, 600, 30),
    ]
    ok = try_configure(cap, combos)
    if not ok:
        cap.release()
        return None
    return cap


def open_source(preferred=PREFERRED_IDX, fallback_path=FALLBACK_VIDEO):
    """
    Abre la fuente: cámara preferida, otras cámaras, o fallback a archivo.
    Devuelve (cap, desc) o (None, None).
    """
    # 1) intento: índice preferido
    cap = open_camera_with_v4l2(preferred)
    if cap:
        return cap, f"camera:{preferred}"

    # 2) otros /dev/video*
    for idx in list_video_indices_linux():
        if idx == preferred:
            continue
        cap = open_camera_with_v4l2(idx)
        if cap:
            return cap, f"camera:{idx}"

    # 3) archivo de respaldo
    if fallback_path and osp.exists(fallback_path):
        cap = cv2.VideoCapture(fallback_path)
        if cap.isOpened():
            # Si FPS es 0 (algunos contenedores), fuerza 25.
            if (cap.get(cv2.CAP_PROP_FPS) or 0) <= 1.0:
                cap.set(cv2.CAP_PROP_FPS, 25.0)
            return cap, f"file:{fallback_path}"

    return None, None


def resize_keep_aspect(img, target_h):
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = float(target_h) / float(h)
    return cv2.resize(img, (int(w * scale), target_h))


def toggle_fullscreen(win_name):
    prop = cv2.getWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN)
    full = int(prop) == 1
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)


def main():
    # --- abrir fuente ---
    cap, desc = open_source()
    if not cap:
        logger.error('No se pudo abrir cámara ni archivo. Revisa /dev/video*, permisos y/o ruta del video.')
        logger.info('Sugerencia: en WSL asegúrate de haber habilitado el device passthrough y que /dev/video* exista. Alternativamente, usa un archivo de video como respaldo en FALLBACK_VIDEO.')
        sys.exit(1)
    logger.info('Fuente abierta: %s', desc)

    # --- inicializar pose (MediaPipe) o usar OpenVINO wrapper si está disponible ---
    pose = None
    if use_openvino and openvino_wrapper is not None:
        logger.info('Using OpenVINO for inference (model=%s). MediaPipe will not be initialized.', openvino_model_path)
    else:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,             # 0/1/2 (2=heavy)
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info('MediaPipe Pose model initialized')

    last_t = time.time()
    miss   = 0
    os.makedirs("frames", exist_ok=True)
    saved_count = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            miss += 1
            if miss >= MAX_MISSES:
                logger.info('Fin de la fuente o timeout prolongado.')
                break
            # pequeña espera antes de reintentar (cámara lenta en WSL)
            time.sleep(0.05)
            continue
        miss = 0

        # Selfie-view
        frame_bgr = cv2.flip(frame_bgr, 1)

        # Redimensiona para mostrar
        frame_bgr = resize_keep_aspect(frame_bgr, TARGET_HEIGHT)

        # Inferencia: OpenVINO wrapper (if presente) o MediaPipe
        try:
            if use_openvino and openvino_wrapper is not None:
                processed_frame, landmarks = openvino_wrapper.infer_and_draw(frame_bgr)
                # processed_frame is the frame with landmarks drawn (OpenVINO wrapper handles drawing)
                frame_bgr = processed_frame
            else:
                # MediaPipe path
                # MediaPipe usa RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                # Dibujo de landmarks
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )
        except Exception:
            logger.exception('Error during inference/drawing. Falling back if possible.')
            # If OpenVINO failed mid-run, try to fallback to MediaPipe (if available)
            if openvino_wrapper is not None and pose is None:
                try:
                    pose = mp_pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    logger.info('Fallback: MediaPipe Pose model initialized after OpenVINO failure')
                except Exception:
                    logger.exception('Failed to initialize MediaPipe during fallback.')

        # FPS
        now = time.time()
        dt  = now - last_t
        if dt > 0:
            fps = 1.0 / dt
            cv2.putText(frame_bgr, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        last_t = now

        # Mostrar
        cv2.imshow(WINDOW_NAME, frame_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):     # ESC o q
            break
        elif k == ord('s'):         # guardar frame
            out = osp.join("frames", f"frame_{saved_count:04d}.jpg")
            try:
                cv2.imwrite(out, frame_bgr)
                logger.info('Guardado: %s', out)
                saved_count += 1
            except Exception:
                logger.exception('Error al guardar frame en %s', out)
        elif k == ord('f'):         # fullscreen toggle
            toggle_fullscreen(WINDOW_NAME)

    # Limpieza
    cap.release()
    if pose is not None:
        try:
            pose.close()
        except Exception:
            logger.exception('Error closing MediaPipe pose object')
    cv2.destroyAllWindows()
    logger.info('Webcam script finished. Frames saved to ./frames if any were captured.')


if __name__ == "__main__":
    main()

"""
INAV_Mavlink.py

Script para Raspberry Pi que recibe un arreglo de 10 etiquetas (strings)
y, dependiendo de cada etiqueta, envía por UART un mensaje MAVLink que
anula canales RC (roll, pitch, throttle, yaw) hacia la controladora INAV.

Requisitos:
  pip install pymavlink

Uso básico:
  python INAV_Mavlink.py --device /dev/ttyUSB0 --baud 115200

Nota de seguridad: probar siempre con hélices desconectadas. Este script
envía `RC_CHANNELS_OVERRIDE` para controlar roll/pitch/throttle/yaw. Ajusta
`LABEL_TO_CONTROL` según tus etiquetas reales.
"""

from pymavlink import mavutil
import time
import argparse
import logging
from typing import List, Tuple, Dict
import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time
import math
import os
import argparse
import logging
from pymavlink import mavutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- 1. CONFIGURACIÓN DEL SISTEMA ---

FEATURE_VECTOR_SIZE = 16
COLOR_TOLERANCE = 30.0  # Tolerancia máxima para autenticación por color.
DEBOUNCE_TIME = 1.0     # Tiempo mínimo entre transiciones de armado/desarmado.
MOVEMENT_SPEED = 0.5    # Velocidad de movimiento del dron (m/s).

# Mapeo de comandos de clase a vectores de velocidad [X, Y, Z]
# X: Adelante/Atrás, Y: Izquierda/Derecha, Z: Arriba/Abajo
COMMAND_MAP = {
	'delante': [MOVEMENT_SPEED, 0, 0],   # Adelante
	'atras':   [-MOVEMENT_SPEED, 0, 0],  # Atrás
	'derecha': [0, MOVEMENT_SPEED, 0],   # Derecha
	'izquierda': [0, -MOVEMENT_SPEED, 0],# Izquierda
	'arriba':  [0, 0, -MOVEMENT_SPEED],  # Arriba
	'abajo':   [0, 0, MOVEMENT_SPEED],   # Abajo
	'trans':   [0, 0, 0],                # Detener/Transición
	'idle':    [0, 0, 0],                # Detener/Idle
}

# Dedo Índice Derecho (20) para muestrear la palma/mano
PALM_LANDMARK_INDEX = 20

LANDMARK_INDICES = {
	'l_shoulder': 11, 'r_shoulder': 12,
	'r_elbow': 14, 'l_elbow': 13,
	'r_wrist': 16, 'l_wrist': 15,
	'r_hip': 24, 'l_hip': 23,
	'r_index': PALM_LANDMARK_INDEX
}
POINTS_OF_INTEREST = [
	'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
	'l_wrist', 'r_wrist', 'l_hip', 'r_hip'
]


# --- 2. FUNCIONES MAVLINK (Reales para ArduPilot) ---

def open_connection(device: str, baud: int = 115200, timeout: float = 10.0) -> mavutil.mavlink_connection:
	logging.info("Abriendo conexión MAVLink en %s a %d bps", device, baud)
	master = mavutil.mavlink_connection(device, baud=baud)
	logging.info("Esperando heartbeat (timeout=%ss)...", timeout)
	try:
		master.wait_heartbeat(timeout=timeout)
		logging.info("Heartbeat recibido: sistema=%s componente=%s", master.target_system, master.target_component)
	except Exception as e:
		logging.warning("No se recibió heartbeat dentro de %ss: %s", timeout, e)
	return master


def send_arm_command(master: mavutil.mavlink_connection, arm: bool):
	if master is None:
		logging.warning("No hay conexión MAVLink abierta para armar/desarmar")
		return
	try:
		logging.info("Enviando comando ARM=%s", arm)
		master.mav.command_long_send(
			master.target_system,
			master.target_component,
			mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
			0,  # confirmation
			1 if arm else 0, 0, 0, 0, 0, 0, 0
		)
	except Exception as e:
		logging.error("Error enviando ARM command: %s", e)


def send_velocity_command_mavlink(master: mavutil.mavlink_connection, vx: float, vy: float, vz: float):
	"""Envía velocidad en el frame LOCAL_NED usando SET_POSITION_TARGET_LOCAL_NED.

	vx,vy,vz en m/s. ArduPilot aplicará velocidades si el vehículo está en modo que lo permita.
	"""
	if master is None:
		logging.warning("No hay conexión MAVLink abierta para enviar velocidad")
		return

	# type_mask: ignorar pos(0..2)=1, usar vel(3..5)=0, ignorar acc(6..8)=1, ignorar yaw(9)=1, yaw_rate(10)=1
	TYPE_MASK_VEL = 1991  # bits para ignorar todo excepto velocidades
	try:
		t = int(round(time.time() * 1000)) & 0xFFFFFFFF
		master.mav.set_position_target_local_ned_send(
			t,
			master.target_system,
			master.target_component,
			mavutil.mavlink.MAV_FRAME_LOCAL_NED,
			TYPE_MASK_VEL,
			0, 0, 0,  # x,y,z positions (ignored)
			float(vx), float(vy), float(vz),  # vx, vy, vz
			0, 0, 0,  # afx, afy, afz (ignored)
			0, 0  # yaw, yaw_rate (ignored)
		)
		logging.debug("Velocidad enviada vx=%.2f vy=%.2f vz=%.2f", vx, vy, vz)
	except Exception as e:
		logging.error("Error enviando velocidad MAVLink: %s", e)


# --- 3. FUNCIONES UTILITY (Mínimas) ---

def standardize_keypoints(pose_landmarks):
	if not pose_landmarks:
		return None
	origin_point = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
	l_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['l_shoulder']]
	r_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
	scale_factor = np.sqrt((l_shoulder.x - r_shoulder.x) ** 2 + (l_shoulder.y - r_shoulder.y) ** 2)
	if scale_factor < 1e-6:
		return None
	normalized_coords = []
	for name in POINTS_OF_INTEREST:
		point = pose_landmarks.landmark[LANDMARK_INDICES[name]]
		final_x = (point.x - origin_point.x) / scale_factor
		final_y = (point.y - origin_point.y) / scale_factor
		normalized_coords.extend([final_x, final_y])
	return np.array(normalized_coords, dtype=np.float32)


def get_hand_color(image, pose_landmarks, landmark_index):
	if not pose_landmarks:
		return None
	point = pose_landmarks.landmark[landmark_index]
	h, w, _ = image.shape
	px, py = int(point.x * w), int(point.y * h)
	sample_size = 5
	y1, y2 = max(0, py - sample_size), min(h, py + sample_size)
	x1, x2 = max(0, px - sample_size), min(w, px + sample_size)
	region = image[y1:y2, x1:x2]
	if region.size == 0:
		return None
	avg_color_bgr = np.mean(region, axis=(0, 1))
	avg_color_rgb = avg_color_bgr[::-1]
	return avg_color_rgb.astype(np.float32)


def color_distance(rgb1, rgb2):
	return math.sqrt((rgb1[0] - rgb2[0]) ** 2 + (rgb1[1] - rgb2[1]) ** 2 + (rgb1[2] - rgb2[2]) ** 2)


# --- 4. CARGA DE MODELO ---

try:
	model = load_model('pose_classifier_model2.h5')
	label_encoder = joblib.load('label_encoder2.pkl')
	class_names = list(label_encoder.classes_)
	logging.info('Modelos Keras/Encoder cargados exitosamente.')
except Exception as e:
	logging.error('ERROR al cargar el modelo o el encoder: %s', e)
	raise


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--mav-device', default='/dev/ttyUSB0', help='Dispositivo MAVLink (ej. /dev/ttyUSB0 o COM3)')
	p.add_argument('--mav-baud', type=int, default=115200, help='Baud rate para MAVLink')
	p.add_argument('--send-commands', action='store_true', help='Permitir enviar comandos al FC (usar con precaución)')
	p.add_argument('--allow-arm', action='store_true', help='Permite enviar comando de armado real')
	p.add_argument('--camera', type=int, default=0, help='Índice de la cámara')
	p.add_argument('--confidence', type=float, default=0.85, help='Umbral de confianza para aceptar predicción')
	return p.parse_args()


def run():
	args = parse_args()

	master = None
	if args.send_commands:
		master = open_connection(args.mav_device, args.mav_baud)

	mp_pose = mp.solutions.pose
	is_armed = False
	arm_toggle_ready = True
	last_arm_disarm_time = 0.0

	authenticated_color_rgb = None
	is_authenticated = False
	auth_captured_once = False

	with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		cap = cv2.VideoCapture(args.camera)
		confidence_threshold = args.confidence
		send_cooldown = 0.2
		last_sent_time = 0
		last_sent_label = None

		logging.info('Iniciando inferencia + MAVLink (CTRL+C para detener)')

		try:
			while cap.isOpened():
				success, image = cap.read()
				if not success:
					time.sleep(0.01)
					continue

				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				image_rgb.flags.writeable = False
				results = pose.process(image_rgb)

				predicted_class = None
				predicted_confidence = 0.0

				if results.pose_landmarks:
					normalized_vector = standardize_keypoints(results.pose_landmarks)
					current_color_rgb = get_hand_color(image, results.pose_landmarks, LANDMARK_INDICES['r_index'])

					if normalized_vector is not None and normalized_vector.shape[0] == FEATURE_VECTOR_SIZE:
						X_input = normalized_vector.reshape(1, FEATURE_VECTOR_SIZE)
						prediction_probs = model.predict(X_input, verbose=0)[0]
						predicted_index = int(np.argmax(prediction_probs))
						predicted_confidence = float(prediction_probs[predicted_index])
						if predicted_confidence > confidence_threshold:
							predicted_class = str(label_encoder.inverse_transform([predicted_index])[0])

					# Arm/Disarm logic based on a specific class 'armar'
					if predicted_class == 'armar' and predicted_confidence > confidence_threshold:
						if arm_toggle_ready and (time.time() - last_arm_disarm_time) > DEBOUNCE_TIME:
							if is_armed:
								is_armed = False
								is_authenticated = False
								authenticated_color_rgb = None
								if args.send_commands and args.allow_arm and master:
									send_arm_command(master, False)
								else:
									logging.info('Desarmar (simulado)')
							else:
								if current_color_rgb is not None:
									authenticated_color_rgb = current_color_rgb
									auth_captured_once = True
									is_armed = True
									is_authenticated = True
									if args.send_commands and args.allow_arm and master:
										send_arm_command(master, True)
									else:
										logging.info('Armar (simulado)')
									logging.info('Color autenticado: %s', authenticated_color_rgb.round(1))
								else:
									logging.warning('No se capturó color de palma; ignorando arm')
							last_arm_disarm_time = time.time()
							arm_toggle_ready = False
					elif predicted_class != 'armar':
						arm_toggle_ready = True

					# Authentication check
					prev_auth = is_authenticated
					if is_armed and auth_captured_once and current_color_rgb is not None and authenticated_color_rgb is not None:
						distance = color_distance(current_color_rgb, authenticated_color_rgb)
						is_authenticated = distance < COLOR_TOLERANCE
						if prev_auth != is_authenticated:
							logging.info('Autenticación: %s (dist=%.1f)', 'EXITOSA' if is_authenticated else 'PERDIDA', distance)

					# If armed+authenticated and label is a movement, send velocity
					now = time.time()
					if is_armed and is_authenticated and predicted_class in COMMAND_MAP:
						if (predicted_class != last_sent_label) or ((now - last_sent_time) > send_cooldown):
							vx, vy, vz = COMMAND_MAP[predicted_class]
							if args.send_commands and master:
								send_velocity_command_mavlink(master, vx, vy, vz)
							else:
								logging.info('Simulado: enviar velocidad %s -> vx=%.2f vy=%.2f vz=%.2f', predicted_class, vx, vy, vz)
							last_sent_label = predicted_class
							last_sent_time = now
					elif is_armed and not is_authenticated and predicted_class in COMMAND_MAP and predicted_class not in ['idle', 'trans']:
						logging.warning("Ignorando comando %s: autenticación fallida", predicted_class)
					elif is_armed and predicted_class not in COMMAND_MAP and predicted_class != 'armar':
						# stop
						if args.send_commands and master:
							send_velocity_command_mavlink(master, 0, 0, 0)
						else:
							logging.info('Simulado: enviar velocidad 0 (stop)')

				time.sleep(0.01)

		except KeyboardInterrupt:
			logging.info('Interrupción por usuario')
		finally:
			cap.release()
			logging.info('Inferencia detenida')


if __name__ == '__main__':
	try:
		run()
	except Exception as e:
		logging.exception('Error inesperado: %s', e)


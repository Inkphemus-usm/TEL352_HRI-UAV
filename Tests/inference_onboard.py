import cv2
import mediapipe as mp
import numpy as np
import joblib
# Importación corregida y esencial para TFLite
from tensorflow.lite.python.interpreter import Interpreter 
import time 
import math
import os

# --- 1. CONFIGURACIÓN DEL SISTEMA ---

FEATURE_VECTOR_SIZE = 16 
COLOR_TOLERANCE = 30.0  
DEBOUNCE_TIME = 1.0     
MOVEMENT_SPEED = 0.5    

# Mapeo de comandos de clase a vectores de velocidad [X, Y, Z]
COMMAND_MAP = {
    'delante': [MOVEMENT_SPEED, 0, 0], 
    'atras':   [-MOVEMENT_SPEED, 0, 0], 
    'derecha': [0, MOVEMENT_SPEED, 0],   
    'izquierda': [0, -MOVEMENT_SPEED, 0],
    'arriba':  [0, 0, -MOVEMENT_SPEED], 
    'abajo':   [0, 0, MOVEMENT_SPEED],   
    'trans':   [0, 0, 0],                
    'idle':    [0, 0, 0],                
}

PALM_LANDMARK_INDEX = 20 

LANDMARK_INDICES = {
    'l_shoulder': 11, 'r_shoulder': 12, 'r_elbow': 14, 'l_elbow': 13,
    'r_wrist': 16, 'l_wrist': 15, 'r_hip': 24, 'l_hip': 23,
    'r_index': PALM_LANDMARK_INDEX 
}
POINTS_OF_INTEREST = [
    'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'l_hip', 'r_hip'
]

# --- 2. FUNCIONES MAVLINK (Mínimas/Simuladas) ---

def arm_disarm_drone(should_arm: bool):
    """Simula el envío del comando de armado/desarmado MAVLink."""
    if should_arm:
        print("=============================")
        print("🚀 MAVLINK: ENVIANDO COMANDO ARMADO")
        print("=============================")
    else:
        print("=============================")
        print("🛑 MAVLINK: ENVIANDO COMANDO DESARMADO")
        print("=============================")

def send_velocity_command(x, y, z):
    """Simula el envío del comando de velocidad MAVLink."""
    if x == 0 and y == 0 and z == 0:
        print("MAVLINK: Velocidad: Detenido (0, 0, 0)")
    else:
        print(f"MAVLINK: Velocidad: X={x:.2f} | Y={y:.2f} | Z={z:.2f}")

# --- 3. FUNCIONES UTILITY (Mínimas) ---

def standardize_keypoints(pose_landmarks):
    """Normaliza y estandariza los keypoints centrándolos en el Hombro Derecho."""
    if not pose_landmarks: return None
    origin_point = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
    l_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['l_shoulder']]
    r_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
    scale_factor = np.sqrt((l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2)
    
    if scale_factor < 1e-6: return None

    normalized_coords = []
    for name in POINTS_OF_INTEREST:
        point = pose_landmarks.landmark[LANDMARK_INDICES[name]]
        final_x = (point.x - origin_point.x) / scale_factor
        final_y = (point.y - origin_point.y) / scale_factor
        normalized_coords.extend([final_x, final_y])
    return np.array(normalized_coords, dtype=np.float32)

def get_hand_color(image, pose_landmarks, landmark_index):
    """Obtiene el color RGB promedio de una pequeña región alrededor del keypoint de la palma."""
    if not pose_landmarks: return None
    point = pose_landmarks.landmark[landmark_index]
    h, w, _ = image.shape
    px, py = int(point.x * w), int(point.y * h)
    
    sample_size = 5 
    y1, y2 = max(0, py - sample_size), min(h, py + sample_size)
    x1, x2 = max(0, px - sample_size), min(w, px + sample_size)
    
    region = image[y1:y2, x1:x2]
    if region.size == 0: return None
    
    avg_color_bgr = np.mean(region, axis=(0, 1))
    avg_color_rgb = avg_color_bgr[::-1]
    return avg_color_rgb.astype(np.float32)

def color_distance(rgb1, rgb2):
    """Calcula la distancia euclidiana entre dos colores RGB (para tolerancia)."""
    return math.sqrt(
        (rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2
    )

# --- 4. CARGA DE MODELO (CORREGIDO) ---

TFLITE_MODEL_PATH = 'pose_classifier_lite.tflite' # Asume este nombre para tu modelo .tflite

try:
    # 1. Cargar el LabelEncoder (necesario para mapear el índice de salida a nombre de clase)
    label_encoder = joblib.load('label_encoder2.pkl')
    class_names = list(label_encoder.classes_)
    
    # 2. Cargar el Intérprete TFLite (CLAVE)
    interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Validación
    if input_details[0]['shape'][1] != FEATURE_VECTOR_SIZE:
        raise ValueError("El modelo TFLite espera un tamaño de entrada diferente al configurado.")
    
    print("✅ Intérprete TFLite y Encoder cargados exitosamente.")
except Exception as e:
    print(f"❌ ERROR al cargar el modelo o el encoder: {e}")
    exit()

# --- 5. BUCLE DE INFERENCIA HEADLESS (CORREGIDO) ---

def run_headless_inference():
    mp_pose = mp.solutions.pose
    
    # Control de Estado del Dron
    is_armed = False 
    arm_toggle_ready = True 
    last_arm_disarm_time = 0.0 

    # Autenticación de Operador
    authenticated_color_rgb = None 
    is_authenticated = False       
    auth_captured_once = False     

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        cap = cv2.VideoCapture(0)
        confidence_threshold = 0.85
        
        print("\n--- INICIANDO INFERENCIA HEADLESS (CTRL+C para detener) ---")

        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            # Preprocesamiento
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            
            predicted_class = None
            
            if results.pose_landmarks:
                normalized_vector = standardize_keypoints(results.pose_landmarks)
                current_color_rgb = get_hand_color(
                    image, results.pose_landmarks, LANDMARK_INDICES['r_index']
                ) 

                # Clasificación de Pose
                if normalized_vector is not None and normalized_vector.shape[0] == FEATURE_VECTOR_SIZE:
                    
                    # Preparar la entrada TFLite: [1, 16] float32
                    X_input = normalized_vector.reshape(1, FEATURE_VECTOR_SIZE).astype(np.float32)
                    
                    # --- INFERENCIA TFLITE (El cambio clave) ---
                    
                    # 1. Asignar el tensor de entrada
                    interpreter.set_tensor(input_details[0]['index'], X_input)
                    
                    # 2. Invocar la inferencia
                    interpreter.invoke()
                    
                    # 3. Obtener el tensor de salida (Probabilidades)
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    
                    prediction_probs = output_data[0] # El array de probabilidades
                    
                    predicted_index = np.argmax(prediction_probs)
                    predicted_confidence = prediction_probs[predicted_index]
                    
                    # Obtener la clase solo si la confianza es alta
                    if predicted_confidence > confidence_threshold:
                        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
                    
                
                # --- LÓGICA DE ARMADO (TOGGLE) Y AUTENTICACIÓN ---
                
                if predicted_class == 'armar' and predicted_confidence > confidence_threshold:
                    
                    if arm_toggle_ready and (time.time() - last_arm_disarm_time) > DEBOUNCE_TIME:
                        
                        if is_armed:
                            # Estaba ARMADO -> Desarmar
                            is_armed = False
                            is_authenticated = False
                            authenticated_color_rgb = None
                            arm_disarm_drone(False)
                        else:
                            # Estaba DESARMADO -> Armar
                            if current_color_rgb is not None:
                                authenticated_color_rgb = current_color_rgb
                                auth_captured_once = True
                                is_armed = True
                                is_authenticated = True
                                arm_disarm_drone(True)
                                print(f"INFO: Color Autenticado: {authenticated_color_rgb.round(1)}")
                            else:
                                print("WARN: No se pudo capturar color, se ignora el comando 'armar'.")

                        last_arm_disarm_time = time.time()
                        arm_toggle_ready = False
                
                elif predicted_class != 'armar':
                    arm_toggle_ready = True
                
                
                # --- VERIFICACIÓN DE COLOR (Contínua) ---

                auth_status = is_authenticated
                if is_armed and auth_captured_once:
                    if current_color_rgb is not None and authenticated_color_rgb is not None:
                        distance = color_distance(current_color_rgb, authenticated_color_rgb)
                        
                        if distance < COLOR_TOLERANCE:
                            is_authenticated = True
                        else:
                            is_authenticated = False 
                            
                    # Reporte de estado de autenticación (solo cuando el estado cambia)
                    if auth_status != is_authenticated:
                        auth_msg = "EXITOSA" if is_authenticated else f"PERDIDA (Dist: {distance:.1f})"
                        print(f"INFO: Autenticación de Operador: {auth_msg}")

                    
                # --- EJECUCIÓN DE COMANDOS DE VUELO MAVLINK ---
                
                if is_armed and is_authenticated and predicted_class in COMMAND_MAP:
                    
                    x, y, z = COMMAND_MAP[predicted_class]
                    send_velocity_command(x, y, z)
                    
                elif is_armed and not is_authenticated:
                    # Dron armado, pero operador no es el correcto. Ignorar comandos de movimiento.
                    if predicted_class in COMMAND_MAP and predicted_class not in ['idle', 'trans']:
                         print(f"WARN: IGNORANDO COMANDO '{predicted_class.upper()}'. Autenticación de color fallida.")
                    
                elif is_armed and predicted_class not in COMMAND_MAP and predicted_class != 'armar':
                    # Si está armado, autenticado, y detecta algo no mapeado, se detiene
                    send_velocity_command(0, 0, 0)
            
            # Pequeña pausa para evitar sobrecargar la CPU/USB de la cámara
            time.sleep(0.01)

    cap.release()
    print("--- INFERENCIA DETENIDA ---")

if __name__ == '__main__':
    try:
        run_headless_inference()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError inesperado: {e}")
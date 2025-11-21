import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time 
import math
import os

# --- 1. CONFIGURACIÓN ---

FEATURE_VECTOR_SIZE = 16 
COLOR_TOLERANCE = 30.0  # Tolerancia máxima (distancia RGB) para aceptar el color.
DEBOUNCE_TIME = 1.0     # Esperar 1.0 segundo entre armar/desarmar

# 💡 Dedo Índice Derecho (20) para muestrear la palma/mano
PALM_LANDMARK_INDEX = 20 

LANDMARK_INDICES = {
    'l_shoulder': 11, 'r_shoulder': 12, 'r_wrist': 16, 
    'l_elbow': 13, 'r_elbow': 14, 'l_wrist': 15, 'r_hip': 24, 'l_hip': 23,
    'r_index': PALM_LANDMARK_INDEX # Añadido para facilitar el acceso
}
POINTS_OF_INTEREST = [
    'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist', 'l_hip', 'r_hip'
]
RELEVANT_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (11, 12), (23, 24)            
]

# --- 2. FUNCIONES UTILITY ---

def standardize_keypoints(pose_landmarks):
    """Normaliza y estandariza los keypoints centrándolos en el Hombro Derecho."""
    if not pose_landmarks: return None
    origin_point = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
    origin_x, origin_y = origin_point.x, origin_point.y
    l_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['l_shoulder']]
    r_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
    scale_factor = np.sqrt((l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2)
    if scale_factor < 1e-6: return None
    normalized_coords = []
    for name in POINTS_OF_INTEREST:
        idx = LANDMARK_INDICES[name]
        point = pose_landmarks.landmark[idx]
        final_x = (point.x - origin_x) / scale_factor
        final_y = (point.y - origin_y) / scale_factor
        normalized_coords.extend([final_x, final_y])
    return np.array(normalized_coords, dtype=np.float32)

def get_hand_color(image, pose_landmarks, landmark_index):
    """Obtiene el color RGB promedio de una pequeña región alrededor del keypoint."""
    if not pose_landmarks: return None, None
    point = pose_landmarks.landmark[landmark_index]
    h, w, _ = image.shape
    px, py = int(point.x * w), int(point.y * h)
    sample_size = 5 
    y1, y2 = max(0, py - sample_size), min(h, py + sample_size)
    x1, x2 = max(0, px - sample_size), min(w, px + sample_size)
    region = image[y1:y2, x1:x2]
    if region.size == 0: return None, (px, py)
    avg_color_bgr = np.mean(region, axis=(0, 1))
    avg_color_rgb = avg_color_bgr[::-1] # Convertir BGR a RGB
    return avg_color_rgb, (px, py)

def color_distance(rgb1, rgb2):
    """Calcula la distancia euclidiana entre dos colores RGB."""
    return math.sqrt(
        (rgb1[0] - rgb2[0])**2 + (rgb1[1] - rgb2[1])**2 + (rgb1[2] - rgb2[2])**2
    )

# --- 3. CARGA DE MODELO ---

try:
    model = load_model('pose_classifier_model2.h5')
    label_encoder = joblib.load('label_encoder2.pkl')
    class_names = list(label_encoder.classes_)
    print("✅ Modelos cargados.")
except Exception as e:
    print(f"❌ ERROR al cargar el modelo o el encoder: {e}")
    exit()

# --- 4. INICIALIZACIÓN DE VARIABLES DE CONTROL Y ESTADO ---

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Control del Dron
is_armed = False 
arm_toggle_ready = True 
last_arm_disarm_time = 0.0 

# Autenticación de Operador
authenticated_color_rgb = None 
is_authenticated = False       
auth_captured_once = False     # Flag para saber si ya se capturó el color inicial

# --- 5. BUCLE PRINCIPAL ---

with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(0)
    current_action = "Esperando pose..."
    confidence_threshold = 0.85

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image_bgr = image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_bgr.flags.writeable = True
        
        normalized_vector = None
        predicted_class = None
        
        # --- LÓGICA DE DETECCIÓN Y CLASIFICACIÓN ---
        
        if results.pose_landmarks:
            normalized_vector = standardize_keypoints(results.pose_landmarks)
            
            # 💡 CAMBIO: Usar el Dedo Índice Derecho (r_index, índice 20) para el muestreo de color
            current_color_rgb, wrist_coords = get_hand_color(
                image_bgr, results.pose_landmarks, LANDMARK_INDICES['r_index']
            ) 

            if normalized_vector is not None and normalized_vector.shape[0] == FEATURE_VECTOR_SIZE:
                
                X_input = normalized_vector.reshape(1, FEATURE_VECTOR_SIZE) 
                prediction_probs = model.predict(X_input, verbose=0)[0]
                predicted_index = np.argmax(prediction_probs)
                predicted_confidence = prediction_probs[predicted_index]
                
                if predicted_confidence > confidence_threshold:
                    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
                    current_action = f"{predicted_class.upper()} ({predicted_confidence*100:.1f}%)"
                else:
                    current_action = "Pose no reconocida..."

            
            # --- GESTIÓN DE ARMADO (TOGGLE) Y AUTENTICACIÓN ---
            
            # Solo si la clase predicha es 'armar' y la confianza es alta
            if predicted_class == 'armar' and predicted_confidence > confidence_threshold:
                
                if arm_toggle_ready and (time.time() - last_arm_disarm_time) > DEBOUNCE_TIME:
                    
                    if is_armed:
                        # Estaba ARMADO -> Desarmar
                        is_armed = False
                        is_authenticated = False 
                        authenticated_color_rgb = None
                        print("🛑 ESTADO: DESARMADO (Desarmado por 'armar')")
                    else:
                        # Estaba DESARMADO -> Armar
                        is_armed = True
                        
                        # --- CAPTURA DE COLOR AL ARMAR (Autenticación) ---
                        if current_color_rgb is not None:
                            authenticated_color_rgb = current_color_rgb
                            auth_captured_once = True
                            is_authenticated = True
                            print(f"🚀 ESTADO: ARMADO. Color Autenticado: {authenticated_color_rgb.round(1)}")
                        else:
                            is_armed = False # Si no se puede capturar color, no se arma.
                            print("❌ No se pudo capturar el color. Falló el armado.")

                    last_arm_disarm_time = time.time()
                    arm_toggle_ready = False
            
            elif predicted_class != 'armar':
                arm_toggle_ready = True
            
            
            # --- VERIFICACIÓN DE COLOR (Contínua) ---
            
            auth_status = "PENDIENTE"
            auth_color = (150, 150, 150) # Gris

            if is_armed and auth_captured_once:
                if current_color_rgb is not None and authenticated_color_rgb is not None:
                    distance = color_distance(current_color_rgb, authenticated_color_rgb)
                    
                    if distance < COLOR_TOLERANCE:
                        is_authenticated = True
                        auth_status = "AUTENTICADO OK"
                        auth_color = (0, 255, 0) # Verde
                    else:
                        # El color se perdió, pero el dron sigue armado.
                        is_authenticated = False 
                        auth_status = f"AUTENTICACIÓN PERDIDA (Dist: {distance:.1f})"
                        auth_color = (0, 0, 255) # Rojo
                
                
            # --- EJECUCIÓN DE COMANDOS DE VUELO ---
            
            if is_armed and is_authenticated and predicted_class not in ['armar', None]:
                # 💡 AQUÍ VA LA LÓGICA MAVLink para comandos de movimiento
                pass 
            elif is_armed and not is_authenticated:
                # Dron armado, pero operador no es el correcto. Ignorar comandos de movimiento.
                
                # CORRECCIÓN DE ERROR: Chequear si predicted_class es None antes de llamar a .upper()
                action_name = predicted_class.upper() if predicted_class else "COMANDO"
                
                current_action = f"IGNORANDO {action_name}. Autenticación fallida."
            
            
            # --- VISUALIZACIÓN ---
            
            # Dibujar pose
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, RELEVANT_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            
            # Dibujar el punto de muestreo y el estado
            if current_color_rgb is not None and wrist_coords is not None:
                cv2.circle(image_bgr, wrist_coords, 10, auth_color, -1)
                cv2.putText(image_bgr, auth_status, (wrist_coords[0] + 15, wrist_coords[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, auth_color, 2)


        # Mostrar estado general del dron y la acción clasificada
        estado_dron = "ARMADO" if is_armed else "DESARMADO"
        text_color = (0, 255, 0) if is_armed else (0, 0, 255)
        
        cv2.putText(
            image_bgr,
            f"ESTADO DRON: {estado_dron}",
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
        )

        cv2.putText(
            image_bgr,
            f"ACCION: {current_action}",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA 
        )

        cv2.imshow('Control HRI con Autenticacion por Color', image_bgr)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
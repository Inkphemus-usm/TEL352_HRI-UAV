import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model


FEATURE_VECTOR_SIZE = 16 

LANDMARK_INDICES = {
    'l_shoulder': 11, 'r_shoulder': 12, # r_shoulder es el origen (0,0)
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24
}

POINTS_OF_INTEREST = [
    'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist',
    'l_hip', 'r_hip'
]

RELEVANT_CONNECTIONS = [
    (11, 13), (13, 15), 
    (12, 14), (14, 16),  
    (11, 23), (12, 24), 
    (11, 12),            
    (23, 24)            
]

def standardize_keypoints(pose_landmarks):
    """Normaliza y estandariza los keypoints centrándolos en el Hombro Derecho."""
    if not pose_landmarks:
        return None

    origin_point = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
    origin_x, origin_y = origin_point.x, origin_point.y

    l_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['l_shoulder']]
    r_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]

    scale_factor = np.sqrt(
        (l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2
    )

    if scale_factor < 1e-6:
        return None

    normalized_coords = []
    for name in POINTS_OF_INTEREST:
        idx = LANDMARK_INDICES[name]
        point = pose_landmarks.landmark[idx]
        
        final_x = (point.x - origin_x) / scale_factor
        final_y = (point.y - origin_y) / scale_factor
        
        normalized_coords.extend([final_x, final_y])

    return np.array(normalized_coords, dtype=np.float32)

try:
    model = load_model('pose_classifier_model.h5')
    print("✅ Modelo Keras cargado exitosamente.")
    
    label_encoder = joblib.load('label_encoder.pkl')
    class_names = list(label_encoder.classes_)
    print(f"✅ LabelEncoder cargado. Clases: {class_names}")

except Exception as e:
    print(f"❌ ERROR al cargar el modelo o el encoder: {e}")
    exit()


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(0)
    
    current_action = "Esperando pose..."
    confidence_threshold = 0.85 # mínimo de confianza para aceptar una predicción
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image_bgr = image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = pose.process(image_rgb)
        
        normalized_vector = None
        
        if results.pose_landmarks:
            normalized_vector = standardize_keypoints(results.pose_landmarks)

            if normalized_vector is not None and normalized_vector.shape[0] == FEATURE_VECTOR_SIZE:
                
                X_input = normalized_vector.reshape(1, FEATURE_VECTOR_SIZE) 
                
                prediction_probs = model.predict(X_input, verbose=0)[0]
                
                predicted_index = np.argmax(prediction_probs)
                predicted_confidence = prediction_probs[predicted_index]
                
                if predicted_confidence > confidence_threshold:
                    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
                    current_action = f"{predicted_class} ({predicted_confidence*100:.1f}%)"
            
                    
                else:
                    current_action = "Pose no reconocida..."

            image_bgr.flags.writeable = True
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, RELEVANT_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # Clasificación en la Pantalla
        text_color = (0, 255, 0) if "%)" in current_action else (0, 0, 255)
        cv2.putText(
            image_bgr,
            f"ACCION: {current_action}",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
        )

        cv2.imshow('Clasificador de Pose en Tiempo Real', image_bgr)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
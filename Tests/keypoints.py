import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

LANDMARK_INDICES = {
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24
}
RELEVANT_CONNECTIONS = [
    (11, 13), (13, 15), 
    (12, 14), (14, 16), 
    (11, 23), (12, 24),
    (11, 12),
    (23, 24)
]

def standardize_keypoints(pose_landmarks):
    if not pose_landmarks:
        return None

    origin_point = pose_landmarks.landmark[LANDMARK_INDICES['r_hip']]
    
    origin_x = origin_point.x
    origin_y = origin_point.y

    l_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['l_shoulder']]
    r_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]

    scale_factor = np.sqrt(
        (l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2
    )

    # limite inferior
    if scale_factor < 1e-6:
        return None

    # keypoint normalizados
    normalized_coords = []

    points_of_interest = [
        'l_shoulder', 'r_shoulder',
        'l_elbow', 'r_elbow',
        'l_wrist', 'r_wrist',
        'l_hip', 'r_hip'
    ]
    
    for name in points_of_interest:
        idx = LANDMARK_INDICES[name]
        point = pose_landmarks.landmark[idx]
        
        # Centrado
        normalized_x = (point.x - origin_x)
        normalized_y = (point.y - origin_y)
        
        # Escalado
        final_x = normalized_x / scale_factor
        final_y = normalized_y / scale_factor
        
        normalized_coords.extend([final_x, final_y])

    # vector resultado (16 elementos: 8 puntos * 2 coordenadas)
    return np.array(normalized_coords, dtype=np.float32)

with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        exit()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False

        results = pose.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        normalized_vector = None
        if results.pose_landmarks:
            normalized_vector = standardize_keypoints(results.pose_landmarks)

            mp_drawing.draw_landmarks(
                image_bgr, 
                RELEVANT_CONNECTIONS, 
                RELEVANT_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3) # Verde para la conexión
            )

        if normalized_vector is not None:
            cv2.putText(
                image_bgr,
                f"Vector Normalizado listo para clasificar. Forma: {normalized_vector.shape}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                image_bgr,
                "Pose no detectada o invalida.",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        cv2.imshow('Deteccion de Gestos (Blazepose + Normalizacion)', image_bgr)
        

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
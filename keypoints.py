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
    # Brazos
    (11, 13), (13, 15),  # Brazo Izquierdo (Hombro-Codo-Muñeca)
    (12, 14), (14, 16),  # Brazo Derecho (Hombro-Codo-Muñeca)
    
    # Torso (Hombros a Caderas)
    (11, 23), (12, 24),
    
    # Tronco Superior (Hombro a Hombro)
    (11, 12),
    
    # Caderas (Cadera a Cadera)
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

    # factor de escala
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
        
        # Centrado: Restar el origen
        normalized_x = (point.x - origin_x)
        normalized_y = (point.y - origin_y)
        
        # Escalado: Dividir por el factor de escala
        final_x = normalized_x / scale_factor
        final_y = normalized_y / scale_factor
        
        normalized_coords.extend([final_x, final_y])

    # vector resultado (16 elementos: 8 puntos * 2 coordenadas)
    return np.array(normalized_coords, dtype=np.float32)

with mp_pose.Pose(
    model_complexity=1, # 0, 1 o 2. Usar 1 (general) o 0 (lite) para más velocidad.
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # Abrir la webcam (0 es el índice de la cámara por defecto)
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        print("Error al abrir la cámara.")
        exit()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Voltear la imagen horizontalmente para un 'efecto espejo' más natural
        image = cv2.flip(image, 1)
        
        # Convertir BGR a RGB, que es el formato de entrada de MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Hacer que la imagen no sea editable (optimización)
        image_rgb.flags.writeable = False

        # Procesar la imagen con MediaPipe Pose
        results = pose.process(image_rgb)
        
        # Volver la imagen editable y RGB a BGR para mostrar
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        normalized_vector = None
        if results.pose_landmarks:
            # 1. Estandarizar Keypoints (código anterior)
            normalized_vector = standardize_keypoints(results.pose_landmarks)

            # 2. Dibujar *solamente* las conexiones relevantes
            mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                # Usar nuestro conjunto de conexiones en lugar de mp_pose.POSE_CONNECTIONS
                RELEVANT_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3) # Verde para la conexión
            )
            # También puedes iterar sobre los puntos que te interesan y dibujarlos
            # sin usar la función draw_landmarks completa.

        # 3. Mostrar el resultado (el vector normalizado)
        if normalized_vector is not None:
            # Aquí es donde el Segundo Modelo (Clasificador) recibiría el vector
            # print("Vector Normalizado (Solo los primeros 6):", normalized_vector[:6])
            
            # Puedes usar este espacio para la lógica del clasificador (SVM, MLP, etc.)
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


        # Mostrar la imagen
        cv2.imshow('Deteccion de Gestos (Blazepose + Normalizacion)', image_bgr)
        
        # Salir al presionar 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
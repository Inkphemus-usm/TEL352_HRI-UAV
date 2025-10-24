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
    (11, 13), (13, 15),  # Brazo izquierdo
    (12, 14), (14, 16),  # Brazo derecho
    (11, 23), (12, 24),  # Torso 
    (11, 12),            # Hombro a hombro
    (23, 24)             # Cadera a cadera
]

def standardize_keypoints(pose_landmarks):
    if not pose_landmarks:
        return None

    origin_point = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]
    
    origin_x = origin_point.x
    origin_y = origin_point.y

    l_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['l_shoulder']]
    r_shoulder = pose_landmarks.landmark[LANDMARK_INDICES['r_shoulder']]

    scale_factor = np.sqrt(
        (l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2
    )

    if scale_factor < 1e-6: # Evitar divisiones por cero
        return None

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
        
        final_x = (point.x - origin_x) / scale_factor
        final_y = (point.y - origin_y) / scale_factor
        
        normalized_coords.extend([final_x, final_y])

    return np.array(normalized_coords, dtype=np.float32)


def draw_normalized_pose(normalized_vector, output_size=(600, 600)):
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    W, H = output_size
    
    center_x, center_y = W // 2, H // 2
    
    VISUAL_SCALE = 100 

    points_normalized = [(normalized_vector[i], normalized_vector[i+1]) 
                         for i in range(0, len(normalized_vector), 2)]

    points_pixels = []
    for norm_x, norm_y in points_normalized:
        inverted_norm_x = -norm_x
        px = int(center_x + inverted_norm_x * VISUAL_SCALE)
        py = int(center_y - norm_y * VISUAL_SCALE) 
        points_pixels.append((px, py))

    normalized_indices_map = {11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 23: 6, 24: 7}
    RELEVANT_CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (11, 12), (23, 24)
    ]
    
    for p1_mp_idx, p2_mp_idx in RELEVANT_CONNECTIONS:
        idx1 = normalized_indices_map.get(p1_mp_idx)
        idx2 = normalized_indices_map.get(p2_mp_idx)
        
        if idx1 is not None and idx2 is not None:
            pt1 = points_pixels[idx1]
            pt2 = points_pixels[idx2]
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)
    
    for i, (px, py) in enumerate(points_pixels):
        color = (0, 255, 0)
        if i == 1: 
            color = (0, 0, 255) # Rojo para el origen (0,0)
        cv2.circle(canvas, (px, py), 5, color, -1)
        
    return canvas

with mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        
        normalized_vector = None
        
        if results.pose_landmarks:
            normalized_vector = standardize_keypoints(results.pose_landmarks)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True # Hacer editable para dibujar

        if normalized_vector is not None:
            normalized_visualization = draw_normalized_pose(normalized_vector)
            cv2.imshow('Vista del Modelo Clasificador (Normalizado)', normalized_visualization)

            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, RELEVANT_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
            cv2.imshow('Vista de la Camara (Solo Puntos Relevantes)', image_bgr)
            
        else:
            cv2.imshow('Vista del Modelo Clasificador (Normalizado)', np.zeros((600, 600, 3), dtype=np.uint8))
            cv2.imshow('Vista de la Camara (Solo Puntos Relevantes)', image_bgr) # Mostrar la cámara aunque no haya pose.

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
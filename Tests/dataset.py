import cv2
import mediapipe as mp
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

DATASET_BASE_DIR = "../../data"
OUTPUT_FILENAME = "dataset.npz"

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

FEATURE_VECTOR_SIZE = len(POINTS_OF_INTEREST) * 2

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

def create_consolidated_dataset(base_dir):
    """
    Recorre subcarpetas de clases, extrae keypoints de imágenes y consolida el dataset.
    """
    all_X = [] # feats
    all_Y = [] # tags
    total_images_processed = 0
    
    with mp_pose.Pose(
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        for class_name in os.listdir(base_dir):
            class_path = os.path.join(base_dir, class_name)
            
            if not os.path.isdir(class_path):
                continue

            print(f"\nProcesando clase: '{class_name}'...")
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, filename)
                    
                    # Leer la imagen
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  Error al cargar imagen: {filename}")
                        continue
                    
                    #  Blazepose
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = pose.process(image_rgb)
                    
                    if results.pose_landmarks:
                        normalized_vector = standardize_keypoints(results.pose_landmarks)
                        
                        if normalized_vector is not None and normalized_vector.shape[0] == FEATURE_VECTOR_SIZE:
                            all_X.append(normalized_vector)
                            all_Y.append(class_name)
                            total_images_processed += 1
                        
            print(f"  -> {len(all_X) - len(all_Y)} muestras añadidas para '{class_name}'.")

    if not all_X:
        print("\nERROR")
        return

    X_data = np.array(all_X)
    Y_data = np.array(all_Y, dtype=object).reshape(-1, 1)

    np.savez_compressed(OUTPUT_FILENAME, X=X_data, Y=Y_data)
    
    print("\n" + "="*50)
    print(f"✅ CONSOLIDACIÓN COMPLETA.")
    print(f"Total de muestras válidas (frames/imágenes): {total_images_processed}")
    print(f"Dataset guardado en: {OUTPUT_FILENAME}")
    print(f"Forma de X (features): {X_data.shape}")
    print(f"Forma de Y (etiquetas): {Y_data.shape}")
    print("="*50)


# --- 4. EJECUCIÓN ---

if __name__ == "__main__":
    if not os.path.exists(DATASET_BASE_DIR):
        print(f"ERROR: El directorio base '{DATASET_BASE_DIR}' no existe. Por favor, créelo y coloque las carpetas de clase dentro.")
    else:
        create_consolidated_dataset(DATASET_BASE_DIR)
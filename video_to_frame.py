import cv2
import os

# --- 1. CONFIGURACIÓN DE RUTAS ---

# Directorio donde se encuentran tus videos MP4
INPUT_VIDEO_DIR = "media/videos/"
# Directorio donde se guardarán las carpetas de imágenes de salida
OUTPUT_DATA_DIR = "data/" 

# --- 2. FUNCIÓN DE EXTRACCIÓN ---

def extract_frames_from_videos(input_dir, output_dir):
    """
    Recorre el directorio de entrada, procesa cada archivo MP4,
    y guarda sus frames como imágenes JPG en una subcarpeta de salida.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio de salida creado: {output_dir}")

    # Iterar sobre todos los archivos en el directorio de entrada
    for filename in os.listdir(input_dir):
        # Asegurarse de que el archivo es un MP4
        if filename.lower().endswith('.mp4'):
            
            video_path = os.path.join(input_dir, filename)
            
            # El nombre de la clase/carpeta es el nombre del video sin la extensión (.mp4)
            class_name = os.path.splitext(filename)[0]
            output_class_dir = os.path.join(output_dir, class_name)
            
            # Crear la carpeta de salida para la clase si no existe
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            
            print(f"\nProcesando video: {filename} -> Guardando en {output_class_dir}")
            
            # Inicializar el objeto de captura de video
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            if not cap.isOpened():
                print(f"❌ ERROR: No se pudo abrir el video en {video_path}")
                continue

            # Bucle para leer cada frame del video
            while True:
                ret, frame = cap.read()
                
                # 'ret' es True si se leyó correctamente el frame
                if not ret:
                    break  # Salir si se llega al final del video
                
                # Construir el nombre del archivo de salida (ej: data/adelante/00000.jpg)
                # Usamos 5 dígitos para un buen ordenamiento (ej: 00000 a 99999)
                frame_filename = os.path.join(output_class_dir, f"{frame_count:05d}.jpg")
                
                # Guardar el frame como imagen JPG
                cv2.imwrite(frame_filename, frame)
                
                frame_count += 1

            # Liberar el objeto de captura
            cap.release()
            print(f"✅ Extracción completa para '{class_name}'. Total de frames: {frame_count}.")

        # else:
        #     print(f"Archivo omitido (no es MP4): {filename}")

# --- 3. EJECUCIÓN ---

if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO_DIR):
        print(f"❌ ERROR CRÍTICO: El directorio de videos de entrada '{INPUT_VIDEO_DIR}' no existe.")
    else:
        extract_frames_from_videos(INPUT_VIDEO_DIR, OUTPUT_DATA_DIR)
        print("\n¡Proceso de troceado de videos finalizado!")
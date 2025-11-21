import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. CONFIGURACIÓN ---
# El vector de características debe ser consistente con la extracción de Blazepose.
FEATURE_VECTOR_SIZE = 16 
MODEL_PATH = 'pose_classifier_model2.h5'
ENCODER_PATH = 'label_encoder2.pkl'
DATA_FILE = 'dataset2.npz' # Usamos el nombre consolidado

# --- 2. CARGA Y PREPROCESAMIENTO DE DATOS ---
def load_and_preprocess_data(data_file):
    """Carga los datos, fusiona las clases ('atras2' -> 'atras') y prepara las etiquetas."""
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_file}. Asegúrate de ejecutar el script de consolidación.")

    data = np.load(data_file, allow_pickle=True)
    X = data['X'] 
    Y_text = data['Y'].flatten() 

    # 💡 FUSIÓN DE CLASES: Reemplazar 'atras2' por 'atras'
    # Usamos np.where para realizar el reemplazo condicional en el array de etiquetas.
    Y_text = np.where(Y_text == 'atras2', 'atras', Y_text)
    print("✅ Fusión de clases: 'atras2' unida a 'atras'.")
    
    # Codificación de Etiquetas (Label Encoding)
    label_encoder = LabelEncoder()
    Y_int = label_encoder.fit_transform(Y_text)
    
    # One-Hot Encoding
    Y = to_categorical(Y_int)
    
    # Clases únicas después de la fusión
    num_classes = Y.shape[1]
    
    print("-" * 40)
    print(f"Clases finales detectadas: {label_encoder.classes_}")
    print(f"Número de clases final: {num_classes}")
    print(f"Tamaño total del dataset: {X.shape[0]} muestras (frames).")
    print("-" * 40)

    # División de Datos (Train/Test)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=Y_int
    )

    return X_train, X_test, Y_train, Y_test, label_encoder, num_classes

# --- 3. DEFINICIÓN Y ENTRENAMIENTO DEL MODELO ---
def build_and_train_model(X_train, X_test, Y_train, Y_test, num_classes):
    """Define la arquitectura del MLP y entrena el modelo."""
    
    # 💡 Arquitectura del Multilayer Perceptron (MLP)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(FEATURE_VECTOR_SIZE,), name='Input_16_Features'), 
        Dense(64, activation='relu', name='Hidden_64'),
        # La capa de salida debe tener el número de clases finales (N)
        Dense(num_classes, activation='softmax', name='Output_Softmax') 
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Usamos una paciencia de 15 para dar más tiempo, aunque 10 es un buen inicio.
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )

    print("\nIniciando entrenamiento del clasificador...")
    history = model.fit(
        X_train, 
        Y_train, 
        epochs=100,
        batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    return model

# --- 4. EJECUCIÓN DEL FLUJO PRINCIPAL ---
if __name__ == "__main__":
    
    # Cargar y dividir los datos
    X_train, X_test, Y_train, Y_test, label_encoder, num_classes = load_and_preprocess_data(DATA_FILE)
    
    print(f"Tamaño de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño de prueba: {X_test.shape[0]} muestras")
    
    # Entrenar el modelo
    model = build_and_train_model(X_train, X_test, Y_train, Y_test, num_classes)
    
    # Evaluación Final
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nPrecisión final en datos de prueba: {accuracy*100:.2f}%")
    
    # Guardado del Modelo y Encoder (para la PC onboard)
    model.save(MODEL_PATH)
    print(f"Modelo guardado exitosamente en: {MODEL_PATH}")

    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"LabelEncoder guardado en: {ENCODER_PATH}")
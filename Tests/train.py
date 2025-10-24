import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# La lista de los 8 puntos de interés en el orden en que se extraen
POINTS_OF_INTEREST = [
    'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist',
    'l_hip', 'r_hip'
]

FEATURE_VECTOR_SIZE = len(POINTS_OF_INTEREST) * 2

data = np.load('dataset.npz', allow_pickle=True)
X = data['X'] 
Y_text = data['Y'].flatten() 

# Codificación de Etiquetas (Label Encoding)
label_encoder = LabelEncoder()
Y_int = label_encoder.fit_transform(Y_text)

Y = to_categorical(Y_int)

num_classes = Y.shape[1]
print(f"Clases detectadas: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=Y_int
)

print(f"Tamaño de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño de prueba: {X_test.shape[0]} muestras")

model = Sequential([
    Dense(128, activation='relu', input_shape=(FEATURE_VECTOR_SIZE,)), 
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

print("\nIniciando entrenamiento...")
history = model.fit(
    X_train, 
    Y_train, 
    epochs=100,
    batch_size=32,
    validation_data=(X_test, Y_test),
    callbacks=[early_stopping],
    verbose=1
)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nPrecisión final en datos de prueba: {accuracy*100:.2f}%")

# modelo que debe estar en la onboard
model_save_path = 'pose_classifier_model.h5'
model.save(model_save_path)
print(f"Modelo guardado exitosamente en: {model_save_path}")

import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')
print("LabelEncoder guardado en: label_encoder.pkl")
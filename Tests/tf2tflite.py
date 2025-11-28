import tensorflow as tf

# Carga tu modelo Keras entrenado
model = tf.keras.models.load_model('pose_classifier_model2.h5')

# Crea el convertidor TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opcional: Optimización para tamaño
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convierte y guarda el archivo .tflite
tflite_model = converter.convert()
with open('pose_classifier_lite.tflite', 'wb') as f:
    f.write(tflite_model)
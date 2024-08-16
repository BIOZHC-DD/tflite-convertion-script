import tensorflow as tf

saved_model_dir = 'bert-tensorflow2-bert-en-uncased-l-10-h-128-a-2-v2'
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("Dostępne urządzenia GPU:", gpus)

if gpus:
    print("✅ TensorFlow korzysta z akceleracji GPU/Metal!")
else:
    print("❌ Brak GPU — model będzie trenował na CPU")
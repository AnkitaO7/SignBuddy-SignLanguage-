import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# === Paths ===
OLD_MODEL_PATH = 'path/to/model/Number/keras_model.h5'
NEW_DATASET_DIR = 'path/to/NewData/Number'
EXPORT_DIR = 'output/RetrainModel/Number'

# === Constants ===
IMG_SIZE = 224
BATCH_SIZE = 16
CLASS_NAMES = sorted(os.listdir(NEW_DATASET_DIR))

# === Load Previous Model ===
print("Loading previous model...")
model = tf.keras.models.load_model(OLD_MODEL_PATH)

# === Data Augmentation & Loading ===
print("Preparing new dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    NEW_DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=CLASS_NAMES
)

val_gen = datagen.flow_from_directory(
    NEW_DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=CLASS_NAMES
)

# === Optional: Fine-tune the base model ===
print("Enabling fine-tuning of base model...")
model.layers[0].trainable = True  # Assumes base model is the first layer

# Compile again after making layers trainable
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Smaller LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Training ===
print("Fine-tuning on new dataset...")
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,  # You can increase this if needed
    callbacks=[early_stop]
)

# === Save updated model ===
print("Saving updated model...")
model.save(f'{EXPORT_DIR}/keras_model_retrained.h5')

# === Save class labels ===
with open(f'{EXPORT_DIR}/labels_retrained.txt', 'w') as f:
    for idx, class_name in enumerate(CLASS_NAMES):
        f.write(f"{idx} {class_name}\n")

# === Convert to TFLite ===
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f'{EXPORT_DIR}/model_retrained.tflite', 'wb') as f:
    f.write(tflite_model)

print("Retraining complete.")

# ===================== IMPORTS =====================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import json
import os

# ===================== PATHS =====================
train_path = r"C:\Users\Ashwin Verma\OneDrive\Desktop\TRAIN\dataset\food-101\train"
val_path   = r"C:\Users\Ashwin Verma\OneDrive\Desktop\TRAIN\dataset\food-101\validation"

# ===================== DATA PREPARATION =====================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# ===================== MODEL =====================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===================== TRAIN =====================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    verbose=1
)

# ===================== SAVE MODEL =====================
model.save("food_recognition_model.h5")
print("✅ Model saved as food_recognition_model.h5")

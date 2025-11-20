import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from tensorflow.keras.models import load_model

model = load_model("food_recognition_model.h5")
print("✅ Model loaded successfully!")

print(len(model.layers))

for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-30:]:
    layer.trainable = True

print("✅ Last 30 layers unfrozen for fine-tuning")

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = r"C:\Users\Ashwin Verma\OneDrive\Desktop\TRAIN\dataset\food-101\train"
val_path   = r"C:\Users\Ashwin Verma\OneDrive\Desktop\TRAIN\dataset\food-101\validation"

train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path,
                                           target_size=(224,224),
                                           batch_size=32,
                                           class_mode='categorical')

val_data   = val_gen.flow_from_directory(val_path,
                                         target_size=(224,224),
                                         batch_size=32,
                                         class_mode='categorical')

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("fine_tuned_food_model.h5",
                             monitor="val_accuracy",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor="val_loss",
                          patience=3,
                          restore_best_weights=True)



# start finetunning

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=8,
    callbacks=[checkpoint, earlystop],
    verbose=1
)

import tensorflow as tf
from tensorflow.keras.models import model_from_json
import h5py
import json

print("TensorFlow version:", tf.__version__)

h5_path = "fine_tuned_food_model.h5"

print("Opening H5 file...")
with h5py.File(h5_path, 'r') as f:
    # Load model_config safely
    model_config = f.attrs.get("model_config")
    
    if model_config is None:
        raise ValueError("No model_config found in the H5 file!")

    # Detect if bytes or str
    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')

    print("Model config loaded.")

    model_config_json = json.loads(model_config)

print("Rebuilding model structure...")
model = model_from_json(json.dumps(model_config_json["config"]))

print("Loading weights from H5...")
model.load_weights(h5_path)

print("Saving as TensorFlow SavedModel format...")
save_path = "food_model_saved"
model.save(save_path)

print(f"✅ Conversion complete. Saved model to: {save_path}")

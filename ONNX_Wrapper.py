# NOTE: This code needs to be executed in your local environment.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tf2onnx # Requires: pip install tf2onnx

# --- 1. Define Model Dimensions (You must check this against your BLS params) ---
# Assuming your BLS has 10 Feature Groups * 8 Features + 6 Enhancement Groups * 3 Enhancements
N_AN_COLS = 10 * 8 + 6 * 3  # = 98
output_dim = 1 # Single regression target

# --- 2. Build the Keras Wrapper Architecture ---
final_feature_input = keras.Input(shape=(N_AN_COLS,), name='input_An_features')

# Separate Wn into weights and bias (the last element of Wn is the bias term)
# Wn is shaped (N_AN_COLS + 1, 1). We assume the bias is the last row.
Wn_weights = model_qat.Wn[:-1].reshape(N_AN_COLS, 1) 
Wn_bias = model_qat.Wn[-1].reshape(-1)        

# Create a single Dense layer representing Wn.
output_layer = layers.Dense(
    units=output_dim, 
    activation='linear', 
    use_bias=True, 
    name='output_Wn_layer'
)(final_feature_input)

keras_model = keras.Model(inputs=final_feature_input, outputs=output_layer)

# --- 3. Inject Trained NumPy Weights (Wn) into Keras Model ---
keras_model.get_layer('output_Wn_layer').set_weights([Wn_weights, Wn_bias])

# --- 4. Export to ONNX ---
onnx_model_path = "bls_qat_final_layer.onnx"

onnx_model, _ = tf2onnx.convert.from_keras(
    keras_model, 
    output_path=onnx_model_path, 
    opset=13 
)

print(f"ONNX Model Exported: {onnx_model_path}")
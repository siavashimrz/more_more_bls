import numpy as np
import os

# --- CONFIGURATION ---
NPZ_FILE = 'BLS_FINAL_CONSTANTS.npz'
OUTPUT_HEADER = 'my_model_constants.h'

def to_c_array(name, data, dtype):
    """Helper to write a C array string."""
    # Flatten the array to 1D for C compatibility
    flat_data = np.array(data).flatten()
    
    size = len(flat_data)
    c_str = f"// Shape: {data.shape}\n"
    c_str += f"const {dtype} {name}[{size}] = {{\n    "
    
    for i, val in enumerate(flat_data):
        if "float" in dtype:
            c_str += f"{val:.8f}f, "
        elif "uint8" in dtype:
            c_str += f"{int(val)}, "
        elif "int8" in dtype:
            c_str += f"{int(val)}, "
        else:
            c_str += f"{val}, "
            
        if (i + 1) % 10 == 0:
            c_str += "\n    "
    
    c_str += "\n};\n\n"
    return c_str

def quantize_weights_to_uint8(weights_float, S, Zp):
    """Quantizes float weights to uint8 using S and Zp."""
    # Formula: q = round(x / S + Zp)
    # Clip to 0-255 for uint8
    w_q = np.round(weights_float / S + Zp)
    w_q = np.clip(w_q, 0, 255).astype(np.uint8)
    return w_q

def main():
    if not os.path.exists(NPZ_FILE):
        print(f"Error: {NPZ_FILE} not found. Please run Step 1 first.")
        return

    print(f"Loading {NPZ_FILE}...")
    data = np.load(NPZ_FILE, allow_pickle=True)

    print(f"Generating {OUTPUT_HEADER}...")
    with open(OUTPUT_HEADER, 'w') as f:
        f.write("// AUTO-GENERATED FILE. DO NOT EDIT.\n")
        f.write("#ifndef MY_MODEL_CONSTANTS_H\n")
        f.write("#define MY_MODEL_CONSTANTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # --- 1. SCALARS & CONSTANTS (Floats) ---
        # These are simple float arrays
        f.write(to_c_array("S_X", data['S_X'], "float"))
        f.write(to_c_array("S_We", data['S_We'], "float"))
        f.write(to_c_array("S_Zn", data['S_Zn'], "float"))
        f.write(to_c_array("S_Wh", data['S_Wh'], "float"))
        f.write(to_c_array("S_Hn", data['S_Hn'], "float"))
        f.write(to_c_array("S_Wn", data['S_Wn'], "float"))
        
        f.write(to_c_array("clip_min", data['clip_min'], "float"))
        f.write(to_c_array("clip_max", data['clip_max'], "float"))
        f.write(to_c_array("scaler_mean", data['scaler_mean'], "float"))
        f.write(to_c_array("scaler_std", data['scaler_std'], "float"))
        f.write(to_c_array("win_mean", data['win_mean'], "float"))
        f.write(to_c_array("win_dist", data['win_dist'], "float"))

        # --- 2. ZERO POINTS (UINT8) ---
        # We cast these to uint8_t
        f.write(to_c_array("Zp_X", data['Zp_X'].astype(np.uint8), "uint8_t"))
        f.write(to_c_array("Zp_We", data['Zp_We'].astype(np.uint8), "uint8_t"))
        f.write(to_c_array("Zp_Zn", data['Zp_Zn'].astype(np.uint8), "uint8_t"))
        f.write(to_c_array("Zp_Wh", data['Zp_Wh'].astype(np.uint8), "uint8_t"))
        f.write(to_c_array("Zp_Hn", data['Zp_Hn'].astype(np.uint8), "uint8_t"))
        f.write(to_c_array("Zp_Wn", data['Zp_Wn'].astype(np.uint8), "uint8_t"))

        # --- 3. WEIGHTS (Quantized to UINT8) ---
        # This is the critical part. We load FLOAT weights and convert to INT C-arrays.

        # A. Feature Weights (We) - List of arrays
        We_list = data['We']
        S_We_list = data['S_We']
        Zp_We_list = data['Zp_We']
        
        for i, We_matrix in enumerate(We_list):
            # Quantize this specific matrix
            We_q = quantize_weights_to_uint8(We_matrix, S_We_list[i], Zp_We_list[i])
            f.write(to_c_array(f"We_{i}_quant", We_q, "uint8_t"))

        # B. Enhancement Weights (Wh) - List of arrays
        Wh_list = data['Wh']
        S_Wh_list = data['S_Wh']
        Zp_Wh_list = data['Zp_Wh']

        for i, Wh_matrix in enumerate(Wh_list):
            Wh_q = quantize_weights_to_uint8(Wh_matrix, S_Wh_list[i], Zp_Wh_list[i])
            f.write(to_c_array(f"Wh_{i}_quant", Wh_q, "uint8_t"))

        # C. Final Weights (Wn) - Single array
        # Note: Wn is 98x1 (or 99x1). We flatten it.
        Wn_float = data['Wn']
        S_Wn = data['S_Wn']
        Zp_Wn = data['Zp_Wn']
        Wn_q = quantize_weights_to_uint8(Wn_float, S_Wn, Zp_Wn)
        
        f.write(to_c_array("Wn_quant", Wn_q, "uint8_t"))

        f.write("#endif // MY_MODEL_CONSTANTS_H\n")

    print("Success! 'my_model_constants.h' has been generated.")

if __name__ == "__main__":
    main()
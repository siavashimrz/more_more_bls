// File: main.c - Main Application Loop and BLS/Drift Logic
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// --- Project Headers ---
// Shared weights and constants (We, Wh, S, Zp)
#include "bls_shared_weights_2.h"
// Mode-specific configurations (Wn, Reference Buffers, KS_LIMIT, MONITORED_FEATURE_IDX)
#include "bls_mode_configs_2.h"
// Your finalized drift detection logic
#include "drift_detector_2.h"
// #include "test_batch_data_2.h"
#include "test_scenario_1_drift.h"

// --- Constants (Derived from bls_mode_configs.h and BLS structure) ---
// Note: These constants should match the generation script's output
#define N_MODES 6
#define INPUT_DIM 16
#define N_FEATURES 10
#define N_FTR_GROUPS 8
#define N_ENHANCEMENTS 6
#define N_ENH_GROUPS 3

#define N_ZN_FEATURES (N_FEATURES * N_FTR_GROUPS)
#define N_HN_FEATURES (N_ENHANCEMENTS * N_ENH_GROUPS)

#define N_WE_PER_GROUP ((INPUT_DIM + 1) * N_FEATURES)         // 17 * 10 = 170
#define N_WH_PER_GROUP ((N_ZN_FEATURES + 1) * N_ENHANCEMENTS) // 81* 6 = 486

// Window size for KS test (defined in bls_mode_configs.h)
#define BUFFER_SIZE 30
#define KS_LIMIT 11

// --- Global State ---
static int8_t current_mode_idx;
// Pointer to the currently active ModeConfig in Flash (or RAM, but conceptually frozen)
static const ModeConfig_t *active_config;

// Buffers for the monitored feature (uint8_t)
static DataValue Monitored_Feature_Buffer_A[BUFFER_SIZE]; // RAM Buffer A (The working buffers)
static DataValue Monitored_Feature_Buffer_B[BUFFER_SIZE]; // RAM Buffer B

// Pointers for the sliding window logic (these pointers will swap)
static DataValue *Old_Buffer_Ptr = Monitored_Feature_Buffer_A;
static DataValue *New_Buffer_Ptr = Monitored_Feature_Buffer_B;
static int new_buffer_count = 0; // Tracks how many samples are in New_Buffer_Ptr

// --- Utility Functions ---

/**
 * @brief Simple activation function (ReLU).
 */
static inline float relu6(float *x)
{
    if (*x < 0.0f)
        *x = 0.0f;
    else if (*x > 6.0f)
        *x = 6.0f;
}

/**
 * @brief Dequantizes a single value: Q = (I - Zp) * S
 */
static inline float dequantize_value(uint8_t I, float S, uint8_t Zp)
{
    return ((float)I - (float)Zp) * S;
}

/**
 * @brief Quantizes a single floating point value: I = round(Q/S + Zp)
 * Clips the result to the valid uint8_t range [0, 255].
 */
static inline uint8_t quantize_value(float Q, float S, uint8_t Zp)
{
    float I = roundf(Q / S + (float)Zp);
    printf(">> %f\t%f\n", Q, I);

    // Clip to uint8 range [0, 255]
    if (I < 0.0f)
        I = 0.0f;
    if (I > 255.0f)
        I = 255.0f;
    return (uint8_t)I;
}

/**
 * @brief Core BLS prediction function using only quantized (uint8_t) weights.
 * @param X_input The single input vector (INPUT_DIM elements).
 * @return The predicted output y.
 */
float predict_bls_quantized(const float *X_input)
{
    // ===============================================================
    // --- Phase 1: Input Quantization and Pre-processing ---
    float X_scaled[INPUT_DIM];
    // +1 for the constant bias term (0.1)
    DataValue X_quant[INPUT_DIM + 1];
    int N_INPUT_BIAS = INPUT_DIM + 1;

    for (int i = 0; i < INPUT_DIM; i++)
    {
        // 1. Scale and Clip Input X
        X_scaled[i] = (X_input[i] - active_config->scaler_mean[i]) / active_config->scaler_std[i];

        if (X_scaled[i] < active_config->clip_min[i])
            X_scaled[i] = active_config->clip_min[i];
        if (X_scaled[i] > active_config->clip_max[i])
            X_scaled[i] = active_config->clip_max[i];

        // 2. Quantize X
        X_quant[i] = quantize_value(X_scaled[i], active_config->S_X[i], active_config->Zp_X[i]);
    }
    // Add bias term (quantized 0.1) using the last scale/zp for consistency
    X_quant[INPUT_DIM] = quantize_value(0.1f, active_config->S_X[INPUT_DIM - 1], active_config->Zp_X[INPUT_DIM - 1]);

    // ==============================================================
    // --- Phase 2: Feature Nodes (Zn) Calculation ---
    float Zn_dequant[N_ZN_FEATURES];

    // Assuming N_FTR_GROUPS = 1 for simplicity
    for (int k = 0; k < INPUT_DIM; k++)
    {
        float Zi_float_sum = 0.0f;

        // We_quant indices: j * INPUT_DIM + k
        for (int j = 0; j < N_INPUT_BIAS; j++)
        {
            float We_dequant = dequantize_value(We_quant[j * INPUT_DIM + k], S_We[k], Zp_We[k]);
            float X_dequant = dequantize_value(X_quant[j], active_config->S_X[j], active_config->Zp_X[j]);
            Zi_float_sum += X_dequant * We_dequant;
        }

        // Apply linear mapping (phi)
        float phi_val = (Zi_float_sum - active_config->win_mean[0]) / active_config->win_dist[0];

        // Final Zn dequantization (for use in Hn and An)
        Zn_dequant[k] = dequantize_value(
            quantize_value(phi_val, active_config->S_Zn[k], active_config->Zp_Zn[k]),
            active_config->S_Zn[k],
            active_config->Zp_Zn[k]);
    }

    // =========================================================
    // --- Phase 3: Enhancement Nodes (Hn) Calculation ---
    float Hn_dequant[N_HN_FEATURES];

    // Prepare Zn_biased_quant (Zn_quant + Bias)
    DataValue Zn_biased_quant[N_ZN_FEATURES + 1];
    int N_ZN_FEATURES_BIAS = N_ZN_FEATURES + 1;
    for (int i = 0; i < N_ZN_FEATURES; i++)
    {
        Zn_biased_quant[i] = quantize_value(Zn_dequant[i], active_config->S_Zn[i], active_config->Zp_Zn[i]);
    }
    Zn_biased_quant[N_ZN_FEATURES] = quantize_value(0.1f, active_config->S_Zn[N_ZN_FEATURES - 1], active_config->Zp_Zn[N_ZN_FEATURES - 1]);

    // Assuming N_ENH_GROUPS = 3
    for (int j = 0; j < N_ENH_GROUPS; j++)
    {
        // Wh_quant base pointer for group j
        const uint8_t *Wh_quant_base = &Wh_quant[j * (N_ZN_FEATURES_BIAS * N_ENHANCEMENTS)];
        const float *S_Wh_j = &S_Wh[j * N_ENHANCEMENTS];
        const uint8_t *Zp_Wh_j = &Zp_Wh[j * N_ENHANCEMENTS];

        for (int k = 0; k < N_ENHANCEMENTS; k++)
        {
            float Hn_float_sum = 0.0f;

            // Wh_quant indices: l * N_ENHANCEMENTS + k
            for (int l = 0; l < N_ZN_FEATURES_BIAS; l++)
            {
                float Wh_dequant = dequantize_value(Wh_quant_base[l * N_ENHANCEMENTS + k], S_Wh_j[k], Zp_Wh_j[k]);

                // Get the correct S and Zp for Zn_biased_quant[l]
                float Zn_dequant_val;
                if (l < N_ZN_FEATURES)
                {
                    Zn_dequant_val = dequantize_value(Zn_biased_quant[l], active_config->S_Zn[l], active_config->Zp_Zn[l]);
                }
                else
                {
                    // This is the bias term
                    Zn_dequant_val = dequantize_value(Zn_biased_quant[l], active_config->S_Zn[N_ZN_FEATURES - 1], active_config->Zp_Zn[N_ZN_FEATURES - 1]);
                }

                Hn_float_sum += Zn_dequant_val * Wh_dequant;
            }

            // Apply activation function
            float Hn_act = relu6(&Hn_float_sum);

            // Final Hn dequantization
            int Hn_idx = j * N_ENHANCEMENTS + k;
            Hn_dequant[Hn_idx] = dequantize_value(
                quantize_value(Hn_act, active_config->S_Hn[Hn_idx], active_config->Zp_Hn[Hn_idx]),
                active_config->S_Hn[Hn_idx],
                active_config->Zp_Hn[Hn_idx]);
        }
    }

    // --- Phase 4: Output Layer (Wn) Calculation ---
    // An = [Zn_dequant | Hn_dequant]
    int N_AN = N_ZN_FEATURES + N_HN_FEATURES;

    float y_pred = 0.0f;
    for (int i = 0; i < N_AN; i++)
    {
        float An_dequant_val;
        if (i < N_ZN_FEATURES)
        {
            An_dequant_val = Zn_dequant[i];
        }
        else
        {
            An_dequant_val = Hn_dequant[i - N_ZN_FEATURES];
        }

        // Get Wn dequantized (index i)
        float Wn_dequant = dequantize_value(active_config->Wn_quant[i], active_config->S_Wn[i], active_config->Zp_Wn[i]);
        y_pred += An_dequant_val * Wn_dequant;
    }

    return y_pred;
}

/**
 * @brief Attempts to find the best mode configuration after drift is detected.
 * @param new_sorted_buffer The current, sorted data buffer.
 * @return The index of the best matching mode (0 to N_MODES-1) or the current mode.
 */
int search_best_mode(DataValue *new_sorted_buffer)
{
    printf("Starting Mode Search...\n");

    for (int i = 0; i < N_MODES; i++)
    {
        // Compare the new buffer against the frozen reference buffer of mode 'i'
        const ModeConfig_t *mode_i = &Modes[i];

        bool drift_detected = ks_drift_detector(
            mode_i->Reference_Buffer, BUFFER_SIZE,
            new_sorted_buffer, BUFFER_SIZE,
            KS_LIMIT);

        printf("Mode %d drift detected: %d\n", i, drift_detected);

        if (!drift_detected)
        {
            current_mode_idx = i;
            break;
        }
    }
    printf("Best match found: Mode %d.\n", current_mode_idx);
}

/**
 * @brief Initializes the system state.
 */
void initialize_system()
{
    // 1. Set the initial active mode configuration (Mode 0)
    current_mode_idx = 0;
    active_config = &Modes[current_mode_idx];

    // 2. Initialize the "Old" buffer with the Reference Data for Mode 0
    // This provides a clean starting point for the first drift check.
    memcpy(Old_Buffer_Ptr, active_config->Reference_Buffer, BUFFER_SIZE * sizeof(DataValue));

    printf("System Initialized to Mode %d. Buffer Size: %d. KS Limit: %d.\n",
           current_mode_idx, BUFFER_SIZE, KS_LIMIT);
}

/**
 * @brief Main function to process incoming data and manage mode switching.
 * @param X_sample The single input data vector (INPUT_DIM elements).
 * @return The prediction for the sample.
 */
void buffer_and_detection(const float *X_sample)
{
    // 2. Quantize the monitored feature for drift detection
    float monitored_feature = X_sample[MONITORED_FEATURE_IDX];

    // Quantize the monitored feature using the global KS scales (S_KS_Global, Zp_KS_Global)
    // DataValue feature_quant = quantize_value(monitored_feature, S_KS_Global, Zp_KS_Global);
    DataValue feature_quant;

    // 3. Quantization (Formula: q = round(x / S + Zp))
    // We use the input's own quantization params (S_X, Zp_X)
    float quant_val = roundf(monitored_feature / S_KS_Global + Zp_KS_Global);

    // 4. Clip to the 8-bit range [0, 255]
    // We use 'uint8_t' so 0-255 is the range.
    if (quant_val > 255.0f)
    {
        feature_quant = 255;
    }
    else if (quant_val < 0.0f)
    {
        feature_quant = 0;
    }
    else
    {
        feature_quant = (uint8_t)quant_val;
    }

    // 3. Store the quantized feature in the New Buffer
    New_Buffer_Ptr[new_buffer_count] = feature_quant;
    new_buffer_count++;

    // 4. Check for Drift when the buffer is full
    if (new_buffer_count == BUFFER_SIZE)
    {
        // The ks_drift_detector will sort New_Buffer_Ptr in-place.
        // Old_Buffer_Ptr is the previous cycle's sorted data.
        bool drift_detected = ks_drift_detector(
            Old_Buffer_Ptr, BUFFER_SIZE,
            New_Buffer_Ptr, BUFFER_SIZE,
            KS_LIMIT);

        if (drift_detected)
        {
            printf("\n--- DRIFT DETECTED from Mode %d ---\n", current_mode_idx);

            // New_Buffer_Ptr is now sorted, use it for mode search
            search_best_mode(New_Buffer_Ptr);
            // CHANGE MODE
            active_config = &Modes[current_mode_idx];
            printf("!!! MODE SWITCHED to Mode %d. !!!\n", current_mode_idx);

            // When mode switches, the "Old" buffer must be re-initialized
            // to the new mode's Reference Buffer for the next continuous check.
            // This ensures the Old Buffer is representative of the NEW model's expected input.
            memcpy(Old_Buffer_Ptr, active_config->Reference_Buffer, BUFFER_SIZE * sizeof(DataValue));
        }
        else
        {
            // No Drift Detected: Efficient Sliding Window Logic
            // The sorted New buffer becomes the Old buffer for the next cycle.
            // We swap the pointers to avoid a data copy.
            DataValue *temp = Old_Buffer_Ptr;
            Old_Buffer_Ptr = New_Buffer_Ptr;
            New_Buffer_Ptr = temp; // New_Buffer_Ptr now points to the location that Old_Buffer_Ptr *just* vacated
        }

        // Reset the counter for the New Buffer (which is now empty)
        new_buffer_count = 0;
        printf("\n--- Cycle Complete ---\n");
    }
}

void quantize_input(const float *in_float, uint8_t *out_uint8)
{
    for (int i = 0; i < INPUT_DIM; i++)
    {
        float scaled;

        // 1. Standardization (Z-score scaling)
        scaled = (in_float[i] - active_config->scaler_mean[i]) / active_config->scaler_std[i];

        // 2. Min/Max Clipping (based on training range)
        scaled = fmaxf(active_config->clip_min[i], fminf(active_config->clip_max[i], scaled));

        // 3. Quantization (Formula: q = round(x / S + Zp))
        // We use the input's own quantization params (S_X, Zp_X)
        float quant_val = roundf(scaled / active_config->S_X[i] + active_config->Zp_X[i]);

        // 4. Clip to the 8-bit range [0, 255]
        // We use 'uint8_t' so 0-255 is the range.
        if (quant_val > 255.0f)
        {
            out_uint8[i] = 255;
        }
        else if (quant_val < 0.0f)
        {
            out_uint8[i] = 0;
        }
        else
        {
            out_uint8[i] = (uint8_t)quant_val;
        }
    }
}

float bls_inference(const float *input_data_float)
{

    // ==================================================================================
    // STEP 1: INPUT PREPROCESSING & QUANTIZATION
    // ==================================================================================

    uint8_t X_q_features[INPUT_DIM];
    quantize_input(input_data_float, X_q_features);

    uint8_t X_q_biased[INPUT_DIM + 1];
    for (int i = 0; i < INPUT_DIM; i++)
    {
        X_q_biased[i] = X_q_features[i];
    }

    // FIX 1: Quantize bias using Zp_X[0]
    // 0.1f is the bias value used in training
    float bias_q_float = 0.1f / active_config->S_X[0] + (float)active_config->Zp_X[0];
    if (bias_q_float > 255.0f)
        bias_q_float = 255.0f;
    if (bias_q_float < 0.0f)
        bias_q_float = 0.0f;
    X_q_biased[INPUT_DIM] = (uint8_t)bias_q_float;

    // ==================================================================================
    // STEP 2: FEATURE NODES (Zn) CALCULATION
    // ==================================================================================

    float Zn_float[N_ZN_FEATURES];
    uint8_t Zn_q[N_ZN_FEATURES];

    for (int g = 0; g < N_FTR_GROUPS; g++)
    {
        int32_t Zi_acc[N_FEATURES];

        for (int n = 0; n < N_FEATURES; n++)
        {
            int32_t acc = 0;

            // Calculate global index for Zp/Scale access
            int global_idx = g * N_FEATURES + n;
            int32_t zp_w_val = (int32_t)Zp_We[global_idx]; // FIX 2: Use global_idx

            for (int i = 0; i < INPUT_DIM; i++)
            {
                int32_t input_val = (int32_t)X_q_biased[i] - (int32_t)active_config->Zp_X[i];
                // We_quant is (17, 10). Access row 'i', col 'n'.
                int32_t weight_val = (int32_t)We_quant[g * N_WE_PER_GROUP + i * N_FEATURES + n] - zp_w_val;
                acc += input_val * weight_val;
            }

            // Bias term (row 16)
            // FIX 3: Subtract Zp_X[0] because that's what we used to quantize it
            int32_t bias_val = (int32_t)X_q_biased[INPUT_DIM] - (int32_t)active_config->Zp_X[0];
            int32_t weight_val_bias = (int32_t)We_quant[g * N_WE_PER_GROUP + INPUT_DIM * N_FEATURES + n] - zp_w_val;
            acc += bias_val * weight_val_bias;

            Zi_acc[n] = acc;

            // De-quantize & Scale
            // FIX 4: Use global_idx for S_We
            float zi_float = (float)Zi_acc[n] * active_config->S_X[0] * S_We[global_idx];

            float zn_val = (zi_float - active_config->win_mean[g]) / active_config->win_dist[g];
            Zn_float[global_idx] = zn_val;

            float quant_temp = roundf(zn_val / active_config->S_Zn[global_idx] + active_config->Zp_Zn[global_idx]);
            if (quant_temp > 255.0f)
                quant_temp = 255.0f;
            if (quant_temp < 0.0f)
                quant_temp = 0.0f;
            Zn_q[global_idx] = (uint8_t)quant_temp;
        }
    }

    // ==================================================================================
    // STEP 3: ENHANCEMENT NODES (Hn) CALCULATION
    // ==================================================================================

    uint8_t Zn_q_biased[N_ZN_FEATURES + 1];
    for (int i = 0; i < N_ZN_FEATURES; i++)
        Zn_q_biased[i] = Zn_q[i];

    // FIX 5: Quantize bias using Zp_Zn[0]
    float bias_zn_q_float = 0.1f / active_config->S_Zn[0] + (float)active_config->Zp_Zn[0];
    if (bias_zn_q_float > 255.0f)
        bias_zn_q_float = 255.0f;
    if (bias_zn_q_float < 0.0f)
        bias_zn_q_float = 0.0f;
    Zn_q_biased[N_ZN_FEATURES] = (uint8_t)bias_zn_q_float;

    float Hn_float[N_HN_FEATURES];
    uint8_t Hn_q[N_HN_FEATURES];

    for (int g = 0; g < N_ENH_GROUPS; g++)
    {
        for (int n = 0; n < N_ENHANCEMENTS; n++)
        {
            int32_t acc = 0;
            int global_idx = g * N_ENHANCEMENTS + n;
            int32_t zp_wh_val = (int32_t)Zp_Wh[global_idx]; // FIX 6: Use global_idx

            for (int i = 0; i < N_ZN_FEATURES; i++)
            {
                int32_t input_val = (int32_t)Zn_q_biased[i] - (int32_t)active_config->Zp_Zn[i];
                // Wh shape: (81, 6).
                int32_t weight_val = (int32_t)Wh_quant[g * N_WH_PER_GROUP + i * N_ENHANCEMENTS + n] - zp_wh_val;
                acc += input_val * weight_val;
            }
            // Bias term (row 80)
            // FIX 7: Subtract Zp_Zn[0]
            int32_t bias_val = (int32_t)Zn_q_biased[N_ZN_FEATURES] - (int32_t)active_config->Zp_Zn[0];
            int32_t weight_val_bias = (int32_t)Wh_quant[g * N_WH_PER_GROUP + N_ZN_FEATURES * N_ENHANCEMENTS + n] - zp_wh_val;
            acc += bias_val * weight_val_bias;

            // De-quantize
            float hn_temp = (float)acc * active_config->S_Zn[0] * S_Wh[global_idx]; // FIX 8: Use global_idx for Scale

            if (hn_temp < 0.0f)
                hn_temp = 0.0f;
            if (hn_temp > 6.0f)
                hn_temp = 6.0f;

            Hn_float[global_idx] = hn_temp;

            float quant_temp = roundf(hn_temp / active_config->S_Hn[global_idx] + active_config->Zp_Hn[global_idx]);
            if (quant_temp > 255.0f)
                quant_temp = 255.0f;
            if (quant_temp < 0.0f)
                quant_temp = 0.0f;
            Hn_q[global_idx] = (uint8_t)quant_temp;
        }
    }

    // ==================================================================================
    // STEP 4: FINAL PREDICTION (An * Wn)
    // ==================================================================================

    float y_final = 0.0f;

    for (int i = 0; i < N_ZN_FEATURES; i++)
    {
        int32_t input_val = (int32_t)Zn_q[i] - (int32_t)active_config->Zp_Zn[i];
        int32_t weight_val = (int32_t)active_config->Wn_quant[i] - (int32_t)active_config->Zp_Wn[0];
        y_final += (float)(input_val * weight_val) * active_config->S_Zn[i] * active_config->S_Wn[0];
    }

    for (int i = 0; i < N_HN_FEATURES; i++)
    {
        int32_t input_val = (int32_t)Hn_q[i] - (int32_t)active_config->Zp_Hn[i];
        int32_t weight_val = (int32_t)active_config->Wn_quant[N_ZN_FEATURES + i] - (int32_t)active_config->Zp_Wn[0];
        y_final += (float)(input_val * weight_val) * active_config->S_Hn[i] * active_config->S_Wn[0];
    }

    return y_final;
}

// Example main loop (replace with your embedded loop/data acquisition)
int main()
{
    initialize_system();

    printf("--- Starting Prediction Loop ---\n");

    float prediction;
    float rmse = 0;
    float mape = 0;

    for (int i = 0; i < N_TEST_SAMPLES; i++)
    {
        if (i == 70)
            printf("\n>>>> mode 1 started\n\n");
        prediction = bls_inference(test_input_batch[i]);
        printf("--> sample %d error:\t%f\n", i, prediction - test_target_batch[i]);

        buffer_and_detection(test_input_batch[i]);

        rmse = rmse + powf((test_target_batch[i] - prediction), 2.0);
        mape = mape + fabsf((test_target_batch[i] - prediction) / test_target_batch[i]);
    }
    rmse = sqrt(rmse / 100.0);
    mape = mape * 100;
    printf("--- Loop Finished ---\n");
    printf("--- RMSE: %f MAPE: %f (percent) ---\n", rmse, mape);

    return 0;
}

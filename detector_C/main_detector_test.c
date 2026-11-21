#include "drift_detector_2.h"
#include "test_scenario_1_drift.h"
#include "bls_mode_configs_2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main()
{
    DataValue New_Buffer_Ptr[N_TEST_SAMPLES];
    bool was_drift;
    DataValue *buff1;
    for (int k = 0; k < N_TEST_SAMPLES; k++)
    {
        float quant_val = roundf(test_input_batch[k][8] / S_KS_Global + Zp_KS_Global);

        if (quant_val > 255.0f)
        {
            New_Buffer_Ptr[k] = 255;
        }
        else if (quant_val < 0.0f)
        {
            New_Buffer_Ptr[k] = 0;
        }
        else
        {
            New_Buffer_Ptr[k] = (uint8_t)quant_val;
        }
    }
    for (int i = 0; i < 4; i++)
    {
        buff1 = New_Buffer_Ptr + (i * 30);
        qsort(buff1, 30, sizeof(DataValue), compare_uint8);
        was_drift = ks_drift_detector(
            buff1, 30,
            (New_Buffer_Ptr + (i + 1) * 30), 30,
            11);
        printf("round %d drift: %s\n", i, was_drift ? "True" : "False");
    }

    return 0;
}
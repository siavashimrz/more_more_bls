// File: drift_detector.c
#include "drift_detector_2.h"
#include <stdlib.h> // For qsort
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

// Helper function for quicksort comparison
int compare_uint8(const void *a, const void *b)
{
    // Cast the void pointers to the expected type (uint8_t)
    uint8_t ua = *(const uint8_t *)a;
    uint8_t ub = *(const uint8_t *)b;

    // Standard comparison logic for ascending order
    if (ua < ub)
        return -1;
    if (ua > ub)
        return 1;
    return 0;
}

/**
 * @brief Performs the Kolmogorov-Smirnov two-sample test on the cumulative counts
 * of two sorted buffers.
 * * @param buf_1_sorted The "Old" reference buffer (MUST be pre-sorted).
 * @param size_1 The size of the old buffer.
 * @param buf_2_unsorted The "New" buffer (will be sorted in-place).
 * @param size_2 The size of the new buffer.
 * @param limit The integer threshold (D_crit numerator) for drift detection.
 * @return true if D-statistic >= limit (Drift detected), false otherwise.
 * * NOTE: The 'buf_2_unsorted' buffer is sorted in-place and is intended to be
 * used as the 'buf_1_sorted' in the next cycle if no drift is detected.
 */
bool ks_drift_detector(
    const DataValue *buf_1_sorted, int size_1,
    DataValue *buf_2_unsorted, int size_2, // Removed 'const' for in-place sorting
    int limit)
{
    // 1. Sort the incoming "New" buffer in-place.
    qsort(buf_2_unsorted, size_2, sizeof(DataValue), compare_uint8);

    // for (int i = 0; i < 30; i++)
    // {
    //     printf("%d\t%d\n", buf_1_sorted[i], buf_2_unsorted[i]);
    // }
    // Initialize pointers and end markers
    // buf_1_sorted is const (Flash Reference or previous Old Buffer)
    const DataValue *old_pointer = buf_1_sorted;
    // buf_2_unsorted is the RAM buffer, now sorted
    const DataValue *new_pointer = buf_2_unsorted;

    // Calculate end pointers for loop termination
    // Note: We cast away the constness of the end pointer for comparison only,
    // as pointer arithmetic on const pointers is safe but less readable for loop control.
    const DataValue *old_end = old_pointer + size_1;
    const DataValue *new_end = new_pointer + size_2;

    int D = 0;         // Maximum absolute difference in counts (The D-statistic numerator)
    int count_old = 0; // Cumulative count for the old buffer
    int count_new = 0; // Cumulative count for the new buffer

    // The main loop traverses both sorted arrays simultaneously.
    while (old_pointer != old_end && new_pointer != new_end)
    {
        // 2. Advance the pointer(s) with the smaller value(s) and update counts.
        if (*new_pointer > *old_pointer)
        {
            // Old value is smaller or equal (since new > old is false, only old is smaller)
            count_old++;
            old_pointer++;
        }
        else if (*new_pointer < *old_pointer)
        {
            // New value is smaller
            count_new++;
            new_pointer++;
        }
        else // *new_pointer == *old_pointer
        {
            // Values are equal, advance both pointers
            count_new++;
            count_old++;
            new_pointer++;
            old_pointer++;
        }

        // 3. CRUCIAL FIX: Check and update the Maximum Difference (D) on every iteration.
        // This ensures the maximum D is correctly captured, even across steps where count
        // difference doesn't change (the 'else' case).
        int current_D = abs(count_new - count_old);

        if (current_D > D)
        {
            D = current_D;
        }

        // 4. Early Exit Optimization
        if (D >= limit)
        {
            return true; // Drift detected
        }
    }

    // The maximum D is always captured inside the loop, so the final check
    // simply returns the result of the maximum D found.
    return D >= limit;
}
import numpy as np
import pandas as pd

def population_stability_index(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    psi = 0
    for i in range(len(expected_perc) - 1):
        exp_pct = np.mean((expected >= expected_perc[i]) & (expected < expected_perc[i+1]))
        act_pct = np.mean((actual >= actual_perc[i]) & (actual < actual_perc[i+1]))

        if exp_pct == 0 or act_pct == 0:
            continue

        psi += (exp_pct - act_pct) * np.log(exp_pct / act_pct)

    return psi
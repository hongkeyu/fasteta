"""
stage2_numpy_numba.py
Cumulative: pure Python replaced by NumPy vectorization + Numba JIT.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def haversine_numba(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    """
    Haversine over full arrays — compiled to machine code by Numba.
    @jit(nopython=True): entire function compiles on first call.
    nopython=True guarantees compiled execution — raises error if
    any Python object is used inside the function.
    """
    n = len(lat1_arr)
    out = np.empty(n, dtype=np.float64)
    R = 6371.0  # Earth radius in km
    for i in range(n):  # this loop runs in machine code — not Python
        phi1 = lat1_arr[i] * np.pi / 180.0
        phi2 = lat2_arr[i] * np.pi / 180.0
        dphi = (lat2_arr[i] - lat1_arr[i]) * np.pi / 180.0
        dlmb = (lon2_arr[i] - lon1_arr[i]) * np.pi / 180.0
        a = (np.sin(dphi / 2.0) ** 2
             + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2)
        out[i] = R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return out


def run_pipeline(df, zone_lookup=None):
    """
    Feature engineering with NumPy vectorization + Numba JIT.
    Improvements over Stage 1:
    - haversine_numba() called ONCE on entire arrays (not n times)
    - All column extractions use .values — numpy array, not pandas row indexing
    - No Python loop over rows — all operations are array-level
    - zone_lookup ignored — no zone features at this stage
    """
    # extract all columns as numpy arrays at once — avoids per-row indexing
    lat1 = df['Restaurant_latitude'].values
    lon1 = df['Restaurant_longitude'].values
    lat2 = df['Delivery_location_latitude'].values
    lon2 = df['Delivery_location_longitude'].values

    # one JIT-compiled call over all n rows — replaces n Python function calls
    distances = haversine_numba(lat1, lon1, lat2, lon2)

    # stack all features into matrix — all numpy, no Python loop
    return np.column_stack([
        distances,
        df['traffic_encoded'].values,
        df['temperature'].values,
        df['humidity'].values,
        df['precipitation'].values,
        df['Delivery_person_Ratings'].values,
        df['vehicle_encoded'].values,
        df['order_encoded'].values,
    ])

"""
stage3_zone.py
Cumulative: Stage 2 + precomputed zone lookup feature.
"""
import numpy as np
from collections import defaultdict
from stage2_numpy_numba import haversine_numba  # reuse compiled function


def coord_to_zone(lat, lon, precision=2):
    """
    Map coordinates to zone key by rounding.
    precision=2 -> ~1.1km grid cells at Indian latitudes (~10-35 degrees N).
    Returns a tuple so it can be used as a dict key (tuples are hashable).
    """
    return (round(lat, precision), round(lon, precision))


def build_zone_lookup(train_df):
    """
    Precompute average delivery time per geographic zone.
    Called ONCE before benchmarking. Time cost is setup cost —
    not included in pipeline timing.

    Why O(1) at inference:
    - This builds {zone_key: avg_target} dict
    - Inference does zone_lookup[key] — one hash lookup = O(1)
    - Alternative (filter DataFrame per row) = O(n) per row = O(n^2) total
    """
    zone_times = defaultdict(list)
    for _, row in train_df.iterrows():
        key = coord_to_zone(
            row['Restaurant_latitude'],
            row['Restaurant_longitude']
        )
        zone_times[key].append(row['TARGET'])
    return {
        zone: sum(times) / len(times)
        for zone, times in zone_times.items()
    }


def run_pipeline(df, zone_lookup=None):
    """
    Feature engineering: Stage 2 + zone lookup feature.
    Improvements over Stage 2:
    - zone_avg_time added as 9th feature column
    - Zone retrieval: O(1) dict lookup per row
    - haversine_numba reused from stage2 — same compiled function
    """
    if zone_lookup is None:
        raise ValueError(
            "stage3 requires zone_lookup. "
            "Call build_zone_lookup(train_df) first."
        )

    lat1 = df['Restaurant_latitude'].values
    lon1 = df['Restaurant_longitude'].values
    lat2 = df['Delivery_location_latitude'].values
    lon2 = df['Delivery_location_longitude'].values

    distances = haversine_numba(lat1, lon1, lat2, lon2)

    # O(1) dict lookup per row — global mean fallback for unseen zones
    global_mean = sum(zone_lookup.values()) / len(zone_lookup)
    zone_times = np.array([
        zone_lookup.get(
            coord_to_zone(lat, lon),
            global_mean  # fallback: zone not seen in training
        )
        for lat, lon in zip(lat1, lon1)
    ])

    return np.column_stack([
        distances,
        df['traffic_encoded'].values,
        df['temperature'].values,
        df['humidity'].values,
        df['precipitation'].values,
        df['Delivery_person_Ratings'].values,
        df['vehicle_encoded'].values,
        df['order_encoded'].values,
        zone_times,  # new feature — added at this stage only
    ])

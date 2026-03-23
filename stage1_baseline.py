"""
stage1_baseline.py
Pure Python pipeline. Intentionally unoptimized. Correctness reference.
"""
import math
import numpy as np


def haversine_single(lat1, lon1, lat2, lon2):
    """
    Haversine distance for one coordinate pair.
    Pure Python — uses math module, not numpy.
    Called once per row in the baseline loop.
    """
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def run_pipeline(df, zone_lookup=None):
    """
    Baseline feature engineering — pure Python loop.
    Intentional inefficiencies (for profiling to expose):
    - haversine_single() called n times (function call overhead x n)
    - df.iloc[i] accesses one row at a time (slow pandas indexing)
    - list.append() inside loop (repeated memory allocation)
    - zone_lookup ignored — no zone features at this stage
    """
    features = []

    for i in range(len(df)):
        row = df.iloc[i]  # slow: one pandas row at a time

        dist = haversine_single(  # function call overhead x n rows
            row['Restaurant_latitude'],
            row['Restaurant_longitude'],
            row['Delivery_location_latitude'],
            row['Delivery_location_longitude']
        )
        features.append([
            dist,
            row['traffic_encoded'],
            row['temperature'],
            row['humidity'],
            row['precipitation'],
            row['Delivery_person_Ratings'],
            row['vehicle_encoded'],
            row['order_encoded'],
        ])
    return np.array(features)  # convert once at the end

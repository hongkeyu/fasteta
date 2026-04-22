"""
step0_cleaning.py
Run once. Downloads data from Kaggle (if needed) and produces clean_data.csv.
"""
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

RAW_DATA = "data/Food_Time_new.csv"


def download_data():
    """Download dataset from Kaggle if not already present."""
    if os.path.exists(RAW_DATA):
        return
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "Raw data not found and kagglehub is not installed.\n"
            "Either:\n"
            "  1) pip install kagglehub   (auto-download), or\n"
            "  2) Download manually from https://www.kaggle.com/datasets/"
            "willianoliveiragibin/food-delivery-time\n"
            f"     and place the CSV at {RAW_DATA}"
        )
    print("Downloading dataset from Kaggle ...")
    path = kagglehub.dataset_download("willianoliveiragibin/food-delivery-time")
    # Kaggle file is named "Food_Time new.csv" (with space)
    src = os.path.join(path, "Food_Time new.csv")
    if not os.path.exists(src):
        # fallback: pick the first csv in the download dir
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {path}")
        src = os.path.join(path, csvs[0])
    os.makedirs("data", exist_ok=True)
    shutil.copy(src, RAW_DATA)
    print(f"Saved to {RAW_DATA}")


def fix_coordinate(value):
    """
    Fix systematically malformed coordinate strings.
    Pattern: "12.972.793" has three dot-separated segments.
    Fix: join segment[0] + '.' + segment[1] + segment[2]
    """
    parts = str(value).strip().split('.')
    if len(parts) == 3:
        return float(parts[0] + '.' + parts[1] + parts[2])
    elif len(parts) == 2:
        return float(value)
    return None


def fix_target(value):
    """
    Fix malformed TARGET column.
    Same pattern as coordinates but may have more segments.
    e.g. "3.816.666.667" -> 3.816666667
    """
    parts = str(value).strip().split('.')
    if len(parts) >= 2:
        return float(parts[0] + '.' + ''.join(parts[1:]))
    return None


def main():
    download_data()
    df = pd.read_csv(RAW_DATA)
    rows_before = len(df)
    print(f"Rows before cleaning: {rows_before}")

    # Fix coordinate columns
    coord_cols = [
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude'
    ]
    for col in coord_cols:
        df[col] = df[col].apply(fix_coordinate)

    # Fix TARGET column
    df['TARGET'] = df['TARGET'].apply(fix_target)

    # Drop rows where any fix returned None
    before_drop = len(df)
    df = df.dropna(subset=coord_cols + ['TARGET'])
    dropped_none = before_drop - len(df)
    print(f"Dropped {dropped_none} rows due to unfixable values")

    # Validate coordinates within India's bounding box
    lat_mask = (
        (df['Restaurant_latitude'].between(6.0, 37.0)) &
        (df['Delivery_location_latitude'].between(6.0, 37.0))
    )
    lon_mask = (
        (df['Restaurant_longitude'].between(68.0, 97.0)) &
        (df['Delivery_location_longitude'].between(68.0, 97.0))
    )
    before_bound = len(df)
    df = df[lat_mask & lon_mask]
    dropped_bounds = before_bound - len(df)
    print(f"Dropped {dropped_bounds} rows due to out-of-bound coordinates")

    # Encode categoricals
    for col, new_col in [
        ('Traffic_Level', 'traffic_encoded'),
        ('Type_of_vehicle', 'vehicle_encoded'),
        ('Type_of_order', 'order_encoded'),
    ]:
        le = LabelEncoder()
        df[new_col] = le.fit_transform(df[col])
        print(f"  Encoded {col}: {list(le.classes_)}")

    # Select output columns (all numeric)
    output_cols = [
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude',
        'traffic_encoded', 'vehicle_encoded', 'order_encoded',
        'temperature', 'humidity', 'precipitation',
        'Delivery_person_Age', 'Delivery_person_Ratings',
        'TARGET'
    ]
    df_out = df[output_cols].copy()

    rows_after = len(df_out)
    print(f"\nRows after cleaning: {rows_after}")
    print(f"Total rows dropped: {rows_before - rows_after}")
    print(f"Lat range: [{df_out['Restaurant_latitude'].min():.4f}, {df_out['Restaurant_latitude'].max():.4f}]")
    print(f"Lon range: [{df_out['Restaurant_longitude'].min():.4f}, {df_out['Restaurant_longitude'].max():.4f}]")

    df_out.to_csv("clean_data.csv", index=False)
    print("Saved clean_data.csv")


if __name__ == '__main__':
    main()

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

PAST_WINDOW = 60       # 60 minutes history
FUTURE_WINDOW = 15     # Lookahead to detect flare
STEP = 1               # Shift window by 1 minute
LABELS = ["no_flare", "B", "C", "M", "X"]

FLUX_PATH = "data/interim/goes_xrs_flux_log.parquet"
FLARE_PATH = "data/raw/hek_flare_events.parquet"

OUTPUT_X = "data/processed/X_flux.npy"
OUTPUT_Y = "data/processed/y_class.npy"
OUTPUT_META = "data/processed/meta_index.csv"


def flare_class_to_int(class_type):
    if class_type == "B":
        return 1
    elif class_type == "C":
        return 2
    elif class_type == "M":
        return 3
    elif class_type == "X":
        return 4
    return 0  # no flare


def main():
    print("🔍 Loading flux and flare data...")
    flux_df = pd.read_parquet(FLUX_PATH)
    flare_df = pd.read_parquet(FLARE_PATH)

    flux_df = flux_df.sort_index()
    flare_df = flare_df.sort_values("start_time")

    print("🔄 Preparing label timeline...")
    flare_df["label"] = flare_df["class_type"].map(flare_class_to_int)
    flare_labels = pd.Series(0, index=flux_df.index)

    for _, row in flare_df.iterrows():
        start = row["start_time"]
        end = start + pd.Timedelta(minutes=FUTURE_WINDOW)
        flare_labels[start:end] = row["label"]

    print("🧠 Building X, y sequences...")
    X, y, index_list = [], [], []

    timestamps = flux_df.index
    flux_values = flux_df["flux_log"].values

    for i in tqdm(range(PAST_WINDOW, len(flux_values) - FUTURE_WINDOW, STEP)):
        past_seq = flux_values[i - PAST_WINDOW:i]
        future_label = flare_labels.iloc[i + FUTURE_WINDOW]

        if np.isnan(past_seq).any():
            continue

        X.append(past_seq)
        y.append(future_label)
        index_list.append(timestamps[i])

    X = np.stack(X)
    y = np.array(y)
    meta_index = pd.DataFrame({"timestamp": index_list, "label": y})

    print(f"✅ Created {len(X)} samples.")

    os.makedirs("data/processed", exist_ok=True)
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    meta_index.to_csv(OUTPUT_META, index=False)

    print(f"📁 Saved: X → {OUTPUT_X}, y → {OUTPUT_Y}, meta → {OUTPUT_META}")


if __name__ == "__main__":
    main()

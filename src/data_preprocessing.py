import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/CICIDS2017")
FILE = DATA_DIR / "CICIDS2017.csv"

SAMPLE_SIZE = 200000


def load_dataset():
    print(f"Loading: {FILE.name}")

    data = pd.read_csv(FILE, low_memory=False)

    print(f"Original shape: {data.shape}")

    return data


def clean_dataset(data):
    print("\nCleaning dataset...")

    # Clean column names
    data.columns = data.columns.str.strip()

    # Replace infinite values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Remove rows containing missing values
    data = data.dropna()

    # Remove duplicate rows
    data = data.drop_duplicates()

    print(f"Shape after cleaning: {data.shape}")

    return data


def create_binary_target(data):
    print("\nCreating binary target...")

    # Normal = 0
    # Any attack = 1
    data["target"] = (
        data["Attack Type"]
        .str.strip()
        .str.lower()
        .apply(lambda x: 0 if x == "normal traffic" else 1)
    )

    # Remove original text label
    data = data.drop(columns=["Attack Type"])

    print("\nBinary class distribution:")
    print(data["target"].value_counts())

    return data


def sample_dataset(data):
    print(f"\nSampling {SAMPLE_SIZE} rows...")

    if len(data) > SAMPLE_SIZE:
        data = data.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    print(f"Final dataset shape: {data.shape}")

    return data


if __name__ == "__main__":

    data = load_dataset()

    data = clean_dataset(data)

    data = create_binary_target(data)

    data = sample_dataset(data)

    output_file = DATA_DIR / "processed_data.csv"

    data.to_csv(
    output_file,
    index=False
    )

    print(f"\nSaved processed dataset to: {output_file}")

    print("\nFinal columns:")
    print(data.columns.tolist())

    print("\nFinal dataset:")
    print(data.head())
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler


TRAIN_FILE = Path("../data/CICIDS2017/processed_data.csv")
CLIENT_DIR = Path("../data/clients")
SCALER_FILE = Path("../data/CICIDS2017/scaler.pkl")

NUM_CLIENTS = 5


def create_scaler():

    print("=" * 60)
    print("CREATING COMMON FEATURE SCALER")
    print("=" * 60)

    print("\nLoading client training data...")

    client_data = []

    for client_id in range(1, NUM_CLIENTS + 1):

        file = (
            CLIENT_DIR /
            f"client_{client_id}.csv"
        )

        data = pd.read_csv(file)

        X = data.drop(
            columns=["target"]
        )

        client_data.append(X)

        print(
            f"Client {client_id}: "
            f"{len(data)} samples loaded"
        )

    # Combine only client training data
    training_data = pd.concat(
        client_data,
        ignore_index=True
    )

    print(
        f"\nTotal training samples: "
        f"{len(training_data)}"
    )

    # Create scaler
    scaler = StandardScaler()

    print("\nFitting StandardScaler...")

    scaler.fit(training_data)

    # Save scaler
    SCALER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print(
        f"\nScaler saved to:"
    )

    print(
        SCALER_FILE
    )

    print("\n" + "=" * 60)
    print("SCALER CREATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    create_scaler()
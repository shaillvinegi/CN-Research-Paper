import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


# Paths
DATA_FILE = Path("../data/CICIDS2017/processed_data.csv")
CLIENT_DIR = Path("../data/clients")
GLOBAL_TEST_FILE = Path("../data/CICIDS2017/global_test.csv")


# Settings
NUM_CLIENTS = 5
TEST_SIZE = 0.10
RANDOM_STATE = 42


def create_clients():

    print("=" * 60)
    print("CREATING FEDERATED LEARNING DATA SPLIT")
    print("=" * 60)

    # Load processed dataset
    print("\nLoading processed dataset...")

    data = pd.read_csv(DATA_FILE)

    print(
        f"Total dataset samples: {len(data)}"
    )

    # Separate features and target
    X = data.drop(
        columns=["target"]
    )

    y = data["target"]

    # Create global train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Recreate dataframes
    train_data = X_train.copy()
    train_data["target"] = y_train.values

    test_data = X_test.copy()
    test_data["target"] = y_test.values

    print("\nGlobal split:")
    print(
        f"Training samples: {len(train_data)}"
    )
    print(
        f"Global test samples: {len(test_data)}"
    )

    # Create directories
    CLIENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GLOBAL_TEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save global test dataset
    test_data.to_csv(
        GLOBAL_TEST_FILE,
        index=False
    )

    print(
        f"\nGlobal test dataset saved to:"
    )

    print(
        GLOBAL_TEST_FILE
    )

    # Shuffle training data
    train_data = train_data.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # Split training data equally among clients
    client_dataframes = [
        train_data.iloc[i::NUM_CLIENTS].reset_index(drop=True)
        for i in range(NUM_CLIENTS)
    ]

    print("\nCreating clients...")

    for i, client_data in enumerate(
        client_dataframes
    ):

        filename = (
            CLIENT_DIR /
            f"client_{i + 1}.csv"
        )

        client_data.to_csv(
            filename,
            index=False
        )

        normal_count = (
            client_data["target"] == 0
        ).sum()

        attack_count = (
            client_data["target"] == 1
        ).sum()

        print(
            f"\nClient {i + 1}"
        )

        print(
            f"Total samples : {len(client_data)}"
        )

        print(
            f"Normal samples: {normal_count}"
        )

        print(
            f"Attack samples: {attack_count}"
        )

        print(
            f"Saved to      : {filename}"
        )

    print("\n" + "=" * 60)
    print("DATA SPLIT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    create_clients()
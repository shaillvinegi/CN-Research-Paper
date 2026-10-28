import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from nn_model import FLModel


class FLClient:

    def __init__(self, client_id, seed=42):

        self.client_id = client_id
        self.seed = seed + client_id

        # --------------------------------------------------
        # REPRODUCIBILITY
        # --------------------------------------------------
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --------------------------------------------------
        # FILE PATHS
        # --------------------------------------------------
        self.file = Path(
            f"../data/clients/client_{client_id}.csv"
        )

        self.scaler_file = Path(
            "../data/CICIDS2017/scaler.pkl"
        )

        print(f"Loading: {self.file}")

        # --------------------------------------------------
        # LOAD DATA
        # --------------------------------------------------
        self.data = pd.read_csv(self.file)

        # --------------------------------------------------
        # LOCAL TRAIN / TEST SPLIT
        # --------------------------------------------------
        split_index = int(
            len(self.data) * 0.8
        )

        train_data = self.data.iloc[
            :split_index
        ]

        test_data = self.data.iloc[
            split_index:
        ]

        self.X_train = train_data.drop(
            columns=["target"]
        )

        self.y_train = train_data["target"]

        self.X_test = test_data.drop(
            columns=["target"]
        )

        self.y_test = test_data["target"]

        # --------------------------------------------------
        # LOAD COMMON SCALER
        # --------------------------------------------------
        print("Loading common scaler...")

        self.scaler = joblib.load(
            self.scaler_file
        )

        # --------------------------------------------------
        # SCALE FEATURES
        # --------------------------------------------------
        X_train_scaled = self.scaler.transform(
            self.X_train
        )

        X_test_scaled = self.scaler.transform(
            self.X_test
        )

        # --------------------------------------------------
        # CONVERT TO PYTORCH TENSORS
        # --------------------------------------------------
        self.X_train_tensor = torch.tensor(
            X_train_scaled,
            dtype=torch.float32
        )

        self.y_train_tensor = torch.tensor(
            self.y_train.values,
            dtype=torch.float32
        ).view(-1, 1)

        self.X_test_tensor = torch.tensor(
            X_test_scaled,
            dtype=torch.float32
        )

        self.y_test_tensor = torch.tensor(
            self.y_test.values,
            dtype=torch.float32
        ).view(-1, 1)

        # --------------------------------------------------
        # CREATE MODEL
        # --------------------------------------------------
        torch.manual_seed(self.seed)

        self.model = FLModel(
            input_size=self.X_train.shape[1]
        )

        # --------------------------------------------------
        # HANDLE CLASS IMBALANCE
        # --------------------------------------------------
        positive_count = self.y_train_tensor.sum()

        negative_count = (
            len(self.y_train_tensor)
            - positive_count
        )

        pos_weight = (
            negative_count /
            positive_count
        )

        print(
            f"Attack samples: "
            f"{int(positive_count)}"
        )

        print(
            f"Normal samples: "
            f"{int(negative_count)}"
        )

        print(
            f"Positive class weight: "
            f"{pos_weight.item():.4f}"
        )

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        # --------------------------------------------------
        # DETERMINISTIC DATALOADER
        # --------------------------------------------------
        dataset = TensorDataset(
            self.X_train_tensor,
            self.y_train_tensor
        )

        generator = torch.Generator()

        generator.manual_seed(
            self.seed
        )

        self.train_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=True,
            generator=generator
        )

    # ======================================================
    # LOCAL TRAINING
    # ======================================================

    def train(self, epochs=2):

        self.model.train()

        # --------------------------------------------------
        # IMPORTANT:
        # Create a fresh optimizer for every FedAvg round.
        # This prevents Adam optimizer state from carrying
        # over between global rounds.
        # --------------------------------------------------
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        for epoch in range(epochs):

            total_loss = 0.0

            for X_batch, y_batch in self.train_loader:

                optimizer.zero_grad()

                outputs = self.model(
                    X_batch
                ).squeeze()

                loss = self.criterion(
                    outputs,
                    y_batch.squeeze()
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            average_loss = (
                total_loss /
                len(self.train_loader)
            )

            print(
                f"Client {self.client_id} "
                f"| Epoch {epoch + 1}/{epochs} "
                f"| Loss: {average_loss:.4f}"
            )

    # ======================================================
    # LOCAL EVALUATION
    # ======================================================

    def evaluate(self):

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                self.X_test_tensor
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).int().numpy().flatten()

        accuracy = accuracy_score(
            self.y_test,
            predictions
        )

        f1 = f1_score(
            self.y_test,
            predictions,
            zero_division=0
        )

        return accuracy, f1


# ==========================================================
# TEST CLIENT
# ==========================================================

if __name__ == "__main__":

    print("=" * 50)
    print("TESTING FEDERATED CLIENT")
    print("=" * 50)

    client = FLClient(
        client_id=1,
        seed=42
    )

    print("\nTraining Client 1...")

    client.train(
        epochs=2
    )

    accuracy, f1 = client.evaluate()

    print("\nClient 1 Results")
    print("----------------------")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print("\n" + "=" * 50)
    print("CLIENT TEST COMPLETED")
    print("=" * 50)
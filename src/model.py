import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_FILE = Path("data/CICIDS2017/processed_data.csv")


def load_data():
    data = pd.read_csv(DATA_FILE)

    X = data.drop(columns=["target"])
    y = data["target"]

    return X, y


def train_model(X_train, y_train):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\nModel Results")
    print("----------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    model, scaler = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        scaler,
        X_test,
        y_test
    )
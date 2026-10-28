import random
import numpy as np
import pandas as pd
import joblib
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--threshold",
    type=float,
    default=0.70,
    help="Threshold for selecting model updates"
)
parser.add_argument(
    "--mode",
    choices=["standard", "selective"],
    default="selective",
    help="Federated learning communication mode"
)
args = parser.parse_args()

THRESHOLD = args.threshold

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from nn_model import FLModel
from client import FLClient


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)


# ==========================================================
# FEDERATED LEARNING SETTINGS
# ==========================================================

NUM_CLIENTS = 5
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2



# ==========================================================
# COMMUNICATION MODE
# ==========================================================

# standard = send every client update
# selective = send only significant updates

COMMUNICATION_MODE = args.mode

# Used only in selective mode
UPDATE_THRESHOLD = THRESHOLD

# ==========================================================
# COMMUNICATION SETTINGS
# ==========================================================

PARAMETER_COUNT = 5505
BYTES_PER_PARAMETER = 4

MODEL_UPDATE_SIZE = (
    PARAMETER_COUNT * BYTES_PER_PARAMETER
)

# Simulated network bandwidth
BANDWIDTH_MBPS = 10


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GLOBAL_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "CICIDS2017"
    / "global_test.csv"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "CICIDS2017"
    / "scaler.pkl"
)


# ==========================================================
# LOAD GLOBAL TEST DATA
# ==========================================================

def load_global_test_data():

    print("\nLoading global test dataset...")

    test_data = pd.read_csv(
        GLOBAL_TEST_FILE
    )

    print(
        f"Global test samples: {len(test_data)}"
    )

    X_test = test_data.drop(
        columns=["target"]
    )

    y_test = test_data["target"]

    scaler = joblib.load(
        SCALER_FILE
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.values,
        dtype=torch.float32
    )

    return X_test_tensor, y_test_tensor


# ==========================================================
# GLOBAL MODEL EVALUATION
# ==========================================================

def evaluate_global_model(
    model,
    X_test,
    y_test
):

    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        probabilities = torch.sigmoid(
            outputs
        )

        predictions = (
            probabilities >= 0.5
        ).int().numpy().flatten()

    y_true = (
        y_test
        .numpy()
        .astype(int)
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return accuracy, precision, recall, f1


# ==========================================================
# CONFUSION MATRIX AND FALSE-POSITIVE RATE
# ==========================================================

def save_confusion_matrix(model, X_test, y_test, filename):
    """Save the final model's confusion matrix and print its FPR.

    Label 0 = Normal traffic; label 1 = Attack traffic.
    """
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        probabilities = torch.sigmoid(outputs)
        predictions = (
            probabilities >= 0.5
        ).int().cpu().numpy().flatten()

    y_true = y_test.cpu().numpy().astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0

    print("\nFinal Confusion Matrix")
    print("----------------------")
    print("Rows = actual class; columns = predicted class")
    print("Class 0 = Normal, Class 1 = Attack")
    print(cm)
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positive Rate:  {fpr:.2f}%")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {filename.stem}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix image saved to: {filename}")
    return fpr


# ==========================================================
# MODEL PARAMETER FUNCTIONS
# ==========================================================

def get_model_parameters(model):

    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def set_model_parameters(
    model,
    parameters
):

    model.load_state_dict(
        parameters
    )


# ==========================================================
# FEDAVG
# ==========================================================

def federated_average(
    client_parameters,
    client_sizes
):

    total_samples = sum(
        client_sizes
    )

    global_parameters = {}

    for key in client_parameters[0]:

        weighted_sum = None

        for parameters, size in zip(
            client_parameters,
            client_sizes
        ):

            weight = (
                size / total_samples
            )

            weighted_value = (
                parameters[key] * weight
            )

            if weighted_sum is None:

                weighted_sum = weighted_value

            else:

                weighted_sum += weighted_value

        global_parameters[key] = weighted_sum

    return global_parameters


# ==========================================================
# UPDATE NORM
# ==========================================================

def calculate_update_norm(
    old_parameters,
    new_parameters
):

    total_squared_norm = 0.0

    for key in old_parameters:

        difference = (
            new_parameters[key]
            - old_parameters[key]
        )

        total_squared_norm += (
            torch.sum(
                difference ** 2
            ).item()
        )

    return total_squared_norm ** 0.5


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("COMMUNICATION-EFFICIENT FEDERATED LEARNING")
    print("=" * 70)

    print(f"\nRandom seed: {SEED}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")

    print(
        f"Communication mode: "
        f"{COMMUNICATION_MODE}"
    )

    if COMMUNICATION_MODE == "selective":

        print(
            f"Update threshold: "
            f"{UPDATE_THRESHOLD}"
        )

    else:

        print(
            "Update threshold: "
            "N/A - all updates transmitted"
        )

    # ======================================================
    # LOAD GLOBAL TEST DATA
    # ======================================================

    X_global_test, y_global_test = (
        load_global_test_data()
    )

    # ======================================================
    # CREATE CLIENTS
    # ======================================================

    print("\nCreating federated clients...")

    clients = [
        FLClient(
            client_id=i,
            seed=SEED
        )
        for i in range(
            1,
            NUM_CLIENTS + 1
        )
    ]

    # ======================================================
    # CREATE GLOBAL MODEL
    # ======================================================

    torch.manual_seed(SEED)

    global_model = FLModel(
        input_size=52
    )

    global_parameters = (
        get_model_parameters(
            global_model
        )
    )

    # ======================================================
    # COMMUNICATION COUNTERS
    # ======================================================

    total_possible_updates = (
        NUM_CLIENTS * NUM_ROUNDS
    )

    updates_sent = 0
    updates_skipped = 0

    # ======================================================
    # STORE RESULTS
    # ======================================================

    round_results = []

    # ======================================================
    # FEDERATED TRAINING
    # ======================================================

    for round_number in range(
        1,
        NUM_ROUNDS + 1
    ):

        print("\n")
        print("=" * 70)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/{NUM_ROUNDS}"
        )

        print("=" * 70)

        client_parameters = []
        client_sizes = []

        # --------------------------------------------------
        # CLIENT TRAINING
        # --------------------------------------------------

        for client in clients:

            print(
                f"\n--- Client {client.client_id} ---"
            )

            # Send current global model
            # to the client
            set_model_parameters(
                client.model,
                global_parameters
            )

            # Save model before local training
            old_parameters = (
                get_model_parameters(
                    client.model
                )
            )

            # Local training
            client.train(
                epochs=LOCAL_EPOCHS
            )

            # Get trained parameters
            new_parameters = (
                get_model_parameters(
                    client.model
                )
            )

            # Calculate update magnitude
            update_norm = (
                calculate_update_norm(
                    old_parameters,
                    new_parameters
                )
            )

            print(
                f"Update norm: "
                f"{update_norm:.6f}"
            )

            # --------------------------------------------------
            # COMMUNICATION DECISION
            # --------------------------------------------------

            if (
                COMMUNICATION_MODE == "standard"
                or update_norm >= UPDATE_THRESHOLD
            ):

                print("Status: SENT")

                client_parameters.append(
                    new_parameters
                )

                client_sizes.append(
                    len(client.X_train)
                )

                updates_sent += 1

            else:

                print("Status: SKIPPED")

                updates_skipped += 1

        # --------------------------------------------------
        # FEDAVG AGGREGATION
        # --------------------------------------------------

        if len(client_parameters) > 0:

            global_parameters = (
                federated_average(
                    client_parameters,
                    client_sizes
                )
            )

            set_model_parameters(
                global_model,
                global_parameters
            )

        else:

            print(
                "\nNo client updates "
                "were transmitted."
            )

        # --------------------------------------------------
        # GLOBAL TEST EVALUATION
        # --------------------------------------------------

        (
            accuracy,
            precision,
            recall,
            f1
        ) = evaluate_global_model(
            global_model,
            X_global_test,
            y_global_test
        )

        # --------------------------------------------------
        # STORE ROUND RESULTS
        # --------------------------------------------------

        round_results.append({

            "round": round_number,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "updates_sent": updates_sent,

            "updates_skipped": updates_skipped
        })

        # --------------------------------------------------
        # PRINT ROUND RESULTS
        # --------------------------------------------------

        print("\nRound Results")
        print("----------------------")

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall: {recall:.4f}"
        )

        print(
            f"F1 Score: {f1:.4f}"
        )

        print(
            f"Updates sent so far: "
            f"{updates_sent}"
        )

        print(
            f"Updates skipped so far: "
            f"{updates_skipped}"
        )

    # ======================================================
    # COMMUNICATION ANALYSIS
    # ======================================================

    standard_communication = (
        total_possible_updates *
        MODEL_UPDATE_SIZE
    )

    actual_communication = (
        updates_sent *
        MODEL_UPDATE_SIZE
    )

    bytes_saved = (
        standard_communication
        - actual_communication
    )

    communication_reduction = (
        bytes_saved /
        standard_communication
    ) * 100

    # ======================================================
    # SIMULATED LATENCY
    # ======================================================

    bandwidth_bits_per_second = (
        BANDWIDTH_MBPS * 1_000_000
    )

    standard_latency = (
        standard_communication * 8
        / bandwidth_bits_per_second
    )

    actual_latency = (
        actual_communication * 8
        / bandwidth_bits_per_second
    )

    latency_saved = (
        standard_latency
        - actual_latency
    )

    latency_reduction = (
        latency_saved /
        standard_latency
    ) * 100

    # ======================================================
    # FINAL METRICS
    # ======================================================

    final_accuracy = (
        round_results[-1]["accuracy"]
    )

    final_precision = (
        round_results[-1]["precision"]
    )

    final_recall = (
        round_results[-1]["recall"]
    )

    final_f1 = (
        round_results[-1]["f1"]
    )

    # ======================================================
    # SAVE FINAL CONFUSION MATRIX
    # ======================================================

    results_dir = PROJECT_ROOT / "experiment_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    if COMMUNICATION_MODE == "standard":
        matrix_filename = results_dir / "confusion_matrix_fedavg.png"
    else:
        matrix_filename = (
            results_dir
            / f"confusion_matrix_selective_threshold_{THRESHOLD:.2f}.png"
        )

    final_fpr = save_confusion_matrix(
        global_model,
        X_global_test,
        y_global_test,
        matrix_filename
    )

    # ======================================================
    # FINAL RESULTS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("FEDERATED LEARNING COMPLETED")
    print("=" * 70)

    print(
        f"\nGlobal test samples: "
        f"{len(y_global_test)}"
    )

    print(
        f"\nTotal possible updates: "
        f"{total_possible_updates}"
    )

    print(
        f"Updates sent: "
        f"{updates_sent}"
    )

    print(
        f"Updates skipped: "
        f"{updates_skipped}"
    )

    print(
        f"Communication reduction: "
        f"{communication_reduction:.2f}%"
    )

    print("\nFinal Global Test Metrics")
    print("-------------------------")

    print(
        f"Accuracy: "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{final_precision:.4f}"
    )

    print(
        f"Recall: "
        f"{final_recall:.4f}"
    )

    print(
        f"F1 Score: "
        f"{final_f1:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{final_fpr:.2f}%"
    )

    # ======================================================
    # COMMUNICATION ANALYSIS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("COMMUNICATION ANALYSIS")
    print("=" * 70)

    print(
        f"\nModel parameters: "
        f"{PARAMETER_COUNT}"
    )

    print(
        f"Model update size: "
        f"{MODEL_UPDATE_SIZE:,} bytes"
    )

    print(
        f"\nStandard FedAvg updates: "
        f"{total_possible_updates}"
    )

    print(
        f"Standard communication: "
        f"{standard_communication:,} bytes"
    )

    print(
        f"\nActual transmitted updates: "
        f"{updates_sent}"
    )

    print(
        f"Actual communication: "
        f"{actual_communication:,} bytes"
    )

    print(
        f"\nBytes saved: "
        f"{bytes_saved:,} bytes"
    )

    print(
        f"Communication reduction: "
        f"{communication_reduction:.2f}%"
    )

    # ======================================================
    # LATENCY ANALYSIS
    # ======================================================

    print("\n")
    print("=" * 70)
    print("SIMULATED COMMUNICATION LATENCY")
    print("=" * 70)

    print(
        f"\nAssumed bandwidth: "
        f"{BANDWIDTH_MBPS} Mbps"
    )

    print(
        f"\nStandard simulated latency: "
        f"{standard_latency:.4f} seconds"
    )

    print(
        f"Actual simulated latency: "
        f"{actual_latency:.4f} seconds"
    )

    print(
        f"\nLatency saved: "
        f"{latency_saved:.4f} seconds"
    )

    print(
        f"Latency reduction: "
        f"{latency_reduction:.2f}%"
    )

    # ======================================================
    # COMPLETION
    # ======================================================

    print("\n")
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
"""
modelling_tuning.py
Hyperparameter tuning dengan manual logging MLflow (Skilled/Menengah).
"""

import os
import pickle
import numpy as np
import mlflow
import mlflow.tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/content/drive/MyDrive/MY PROJECT/Hackathon x Digdaya 2026/Membangun Sistem Machine Learning/preprocessing",
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_output")
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
BATCH_SIZE = 64
EPOCHS = 15
RANDOM_SEED = 42

mlflow.set_tracking_uri("mlruns")


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, encoder


def build_model(embedding_dim=128, lstm_units=64, learning_rate=1e-3, num_classes=2):
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, embedding_dim, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(lstm_units)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(y_true, y_pred, filepath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def train_and_log(params, X_train, y_train, X_val, y_val, X_test, y_test, run_name):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = build_model(
            embedding_dim=params["embedding_dim"],
            lstm_units=params["lstm_units"],
            learning_rate=params["learning_rate"],
            num_classes=params["num_classes"],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0),
        ]

        history = model.fit(
            X_train, y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0,
        )

        val_acc = max(history.history["val_accuracy"])
        val_loss = min(history.history["val_loss"])
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_loss", val_loss)

        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        test_accuracy = np.mean(y_pred == y_test)
        test_precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        test_recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        mlflow.tensorflow.log_model(model, "model")

        report = classification_report(y_test, y_pred, target_names=["Negatif", "Positif"], zero_division=0)
        report_path = os.path.join(MODEL_DIR, f"classification_report_{run_name}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        cm_path = os.path.join(MODEL_DIR, f"confusion_matrix_{run_name}.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        model.save(os.path.join(MODEL_DIR, f"model_{run_name}.keras"))

        print(f"[{run_name}] Test Acc: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        return test_accuracy, model


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = load_data()
    num_classes = len(encoder.classes_)
    print(f"Data loaded. Classes: {encoder.classes_}")

    param_grid = [
        {"embedding_dim": 128, "lstm_units": 64, "learning_rate": 1e-3, "num_classes": num_classes, "batch_size": 64, "epochs": EPOCHS},
        {"embedding_dim": 128, "lstm_units": 128, "learning_rate": 1e-3, "num_classes": num_classes, "batch_size": 64, "epochs": EPOCHS},
        {"embedding_dim": 256, "lstm_units": 64, "learning_rate": 5e-4, "num_classes": num_classes, "batch_size": 64, "epochs": EPOCHS},
    ]

    best_acc = 0
    best_run = None

    for i, params in enumerate(param_grid):
        run_name = f"bilstm_tuning_{i+1}"
        acc, _ = train_and_log(params, X_train, y_train, X_val, y_val, X_test, y_test, run_name)
        if acc > best_acc:
            best_acc = acc
            best_run = run_name

    print(f"\nBest run: {best_run} with test accuracy: {best_acc:.4f}")
    print("Check MLflow UI: mlflow ui")


if __name__ == "__main__":
    main()

"""
Training dan penyimpanan model B-Care.
Model: SVM Linear + Firefly Algorithm + Fisher Score
Skenario: FA + FS skenario 5 (n_fireflies=5, max_iter=15)

Cara pakai:
1. Jalankan: python train_save_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. Konfigurasi skenario terbaik untuk aplikasi
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
TOP_K = 10

# Skenario terbaik SVM + FA + FS pada TA: skenario 5
N_FIREFLIES = 5
MAX_ITER = 15

# Rentang parameter SVM yang dioptimasi menggunakan FA
LR_RANGE = (0.0001, 0.001)
LAMBDA_RANGE = (0.0001, 0.01)
C_RANGE = (0.1, 50)

EPOCHS_FINAL = 500
EPOCHS_FITNESS = 100
K_FOLD = 5


# ============================================================
# 2. Load dan preprocessing dataset
# ============================================================
def load_dataset():
    candidate_paths = [
        "/dataset/data.csv",
        "data.csv",
        "./dataset/data.csv",
    ]

    data_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            "Dataset tidak ditemukan. Letakkan file data.csv pada folder dataset/data.csv "
            "atau pada folder yang sama dengan file train_save_model.py."
        )

    df = pd.read_csv(data_path)

    # Drop kolom yang tidak dipakai
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Encoding label diagnosis: M = 1, B = -1
    if "diagnosis" not in df.columns:
        raise ValueError("Kolom 'diagnosis' tidak ditemukan pada dataset.")

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": -1})

    if df["diagnosis"].isna().any():
        raise ValueError("Label diagnosis harus berisi nilai M dan B.")

    feature_names = df.drop(columns=["diagnosis"]).columns.tolist()
    X = df[feature_names].values.astype(float)
    y = df["diagnosis"].values.astype(int)

    return df, X, y, feature_names


# ============================================================
# 3. Normalisasi Min-Max
# ============================================================
def fit_minmax(X_train):
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    return min_vals, max_vals


def transform_minmax(X, min_vals, max_vals):
    denominator = max_vals - min_vals
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return (X - min_vals) / denominator


# ============================================================
# 4. Fisher Score untuk feature selection
# ============================================================
def fisher_score(X, y):
    classes = np.unique(y)
    scores = []

    for i in range(X.shape[1]):
        overall_mean = np.mean(X[:, i])
        numerator = 0.0
        denominator = 0.0

        for c in classes:
            X_c = X[y == c, i]
            if len(X_c) == 0:
                continue

            mean_c = np.mean(X_c)
            var_c = np.var(X_c)
            numerator += len(X_c) * (mean_c - overall_mean) ** 2
            denominator += len(X_c) * var_c

        score = numerator / (denominator + 1e-8)
        scores.append(score)

    return np.array(scores)


# ============================================================
# 5. SVM Linear manual
# ============================================================
def svm_train(
    X,
    y,
    lr=0.001,
    C=0.1,
    lambda_param=0.001,
    epochs=500,
    early_stop=True,
    tolerance=1e-6,
):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        margins = y * (np.dot(X, w) + b)
        hinge_loss = np.maximum(0, 1 - margins)
        loss = lambda_param * np.dot(w, w) + C * np.mean(hinge_loss)
        losses.append(loss)

        dw = 2 * lambda_param * w
        db = 0.0

        misclassified = margins < 1
        if np.any(misclassified):
            dw -= C * np.mean(y[misclassified, None] * X[misclassified], axis=0)
            db -= C * np.mean(y[misclassified])

        w -= lr * dw
        b -= lr * db

        if early_stop and epoch > 0:
            if abs(losses[-2] - losses[-1]) < tolerance:
                break

    return w, b, losses


def svm_decision_function(X, w, b):
    return np.dot(X, w) + b


def svm_predict(X, w, b):
    scores = svm_decision_function(X, w, b)
    preds = np.sign(scores)
    preds[preds == 0] = 1
    return preds.astype(int)


# ============================================================
# 6. Evaluasi model
# ============================================================
def evaluate_model(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 7. Fitness FA menggunakan k-fold F1-score
# ============================================================
def kfold_fitness(X, y, lr, lambda_param, C):
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_k, X_val_k = X[train_idx], X[val_idx]
        y_train_k, y_val_k = y[train_idx], y[val_idx]

        w, b, _ = svm_train(
            X_train_k,
            y_train_k,
            lr=lr,
            C=C,
            lambda_param=lambda_param,
            epochs=EPOCHS_FITNESS,
            early_stop=True,
        )

        y_pred_k = svm_predict(X_val_k, w, b)
        metrics = evaluate_model(y_val_k, y_pred_k)
        f1_scores.append(metrics["f1_score"])

    return float(np.mean(f1_scores))


# ============================================================
# 8. Firefly Algorithm untuk optimasi parameter SVM
# ============================================================
def firefly_algorithm(
    X,
    y,
    n_fireflies=N_FIREFLIES,
    max_iter=MAX_ITER,
    alpha=0.2,
    beta0=1.0,
    gamma=1.0,
):
    np.random.seed(RANDOM_STATE)

    fireflies = np.random.rand(n_fireflies, 3)
    fireflies[:, 0] = fireflies[:, 0] * (LR_RANGE[1] - LR_RANGE[0]) + LR_RANGE[0]
    fireflies[:, 1] = fireflies[:, 1] * (LAMBDA_RANGE[1] - LAMBDA_RANGE[0]) + LAMBDA_RANGE[0]
    fireflies[:, 2] = fireflies[:, 2] * (C_RANGE[1] - C_RANGE[0]) + C_RANGE[0]

    fitness = np.array([
        kfold_fitness(X, y, lr=f[0], lambda_param=f[1], C=f[2])
        for f in fireflies
    ])

    best_idx = np.argmax(fitness)
    global_best = fireflies[best_idx].copy()
    global_best_fitness = fitness[best_idx]
    history = [global_best_fitness]

    for iteration in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] > fitness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    random_step = alpha * (np.random.rand(3) - 0.5)

                    fireflies[i] = (
                        fireflies[i]
                        + beta * (fireflies[j] - fireflies[i])
                        + random_step
                    )

                    fireflies[i, 0] = np.clip(fireflies[i, 0], *LR_RANGE)
                    fireflies[i, 1] = np.clip(fireflies[i, 1], *LAMBDA_RANGE)
                    fireflies[i, 2] = np.clip(fireflies[i, 2], *C_RANGE)

                    fitness[i] = kfold_fitness(
                        X,
                        y,
                        lr=fireflies[i, 0],
                        lambda_param=fireflies[i, 1],
                        C=fireflies[i, 2],
                    )

        best_idx = np.argmax(fitness)
        if fitness[best_idx] > global_best_fitness:
            global_best = fireflies[best_idx].copy()
            global_best_fitness = fitness[best_idx]

        history.append(global_best_fitness)
        alpha *= 0.95
        print(f"Iterasi {iteration + 1}/{max_iter} | Best Fitness: {global_best_fitness:.4f}")

    return global_best, history


# ============================================================
# 9. Main training pipeline
# ============================================================
def main():
    df, X, y, feature_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    min_vals, max_vals = fit_minmax(X_train)
    X_train_norm = transform_minmax(X_train, min_vals, max_vals)
    X_test_norm = transform_minmax(X_test, min_vals, max_vals)

    scores_fs = fisher_score(X_train_norm, y_train)
    selected_idx = np.argsort(scores_fs)[::-1][:TOP_K]
    selected_features = [feature_names[i] for i in selected_idx]

    X_train_fs = X_train_norm[:, selected_idx]
    X_test_fs = X_test_norm[:, selected_idx]

    print("Selected feature indices:", selected_idx)
    print("Selected features:", selected_features)

    best_params, history = firefly_algorithm(X_train_fs, y_train)
    best_lr, best_lambda, best_C = best_params

    print("\n===== PARAMETER TERBAIK FA + FS =====")
    print(f"Best LR     : {best_lr:.6f}")
    print(f"Best Lambda : {best_lambda:.6f}")
    print(f"Best C      : {best_C:.6f}")

    w, b, losses = svm_train(
        X_train_fs,
        y_train,
        lr=best_lr,
        C=best_C,
        lambda_param=best_lambda,
        epochs=EPOCHS_FINAL,
        early_stop=True,
    )

    y_pred = svm_predict(X_test_fs, w, b)
    decision_scores = svm_decision_function(X_test_fs, w, b)
    metrics = evaluate_model(y_test, y_pred)

    try:
        auc = roc_auc_score((y_test == 1).astype(int), decision_scores)
    except Exception:
        auc = None

    metrics["auc"] = auc

    print("\n===== HASIL EVALUASI TEST SET =====")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    artifacts = {
        "app_name": "B-Care",
        "model_name": "SVM + FA + FS",
        "scenario": "Skenario 5: n_fireflies=5, max_iter=15",
        "w": w,
        "b": b,
        "min_vals": min_vals,
        "max_vals": max_vals,
        "selected_idx": selected_idx,
        "selected_features": selected_features,
        "feature_names": feature_names,
        "fisher_scores": scores_fs,
        "best_params": {
            "learning_rate": float(best_lr),
            "lambda": float(best_lambda),
            "C": float(best_C),
        },
        "history_fitness": history,
        "losses": losses,
        "metrics": metrics,
    }

    os.makedirs("model", exist_ok=True)
    model_path = "model/bcare_svm_fa_fs.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(artifacts, file)

    print(f"\nModel berhasil disimpan ke: {model_path}")


if __name__ == "__main__":
    main()

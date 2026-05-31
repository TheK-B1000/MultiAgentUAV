import argparse
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def main():
    parser = argparse.ArgumentParser(description="Audit q_phi context signal via classification probe.")
    parser.add_argument("--csv", type=str, required=True, help="Path to E3 step telemetry CSV file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    print(f"Loading telemetry from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows.")

    # Find context columns
    ctx_cols = [f"q_phi_context_{i}" for i in range(95)]
    missing_cols = [col for col in ctx_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing context columns in CSV: {len(missing_cols)} columns missing.")
        sys.exit(1)

    # 1. Extract features
    X_raw = df[ctx_cols].values.astype(np.float32)

    # 2. Extract targets
    # opponent_id (integer)
    if "opponent_id" in df.columns:
        # filter out rows where opponent_id is invalid/missing (e.g. -1)
        y_opp = df["opponent_id"].values.astype(np.int64)
    else:
        y_opp = None

    # phase_id (integer, 0..5)
    if "phase_id" in df.columns:
        y_phase = df["phase_id"].values.astype(np.int64)
    else:
        y_phase = None

    # score_outcome ("loss", "draw", "win") -> map to integers 0, 1, 2
    if "score_outcome" in df.columns:
        outcome_map = {"loss": 0, "draw": 1, "win": 2}
        y_outcome = df["score_outcome"].map(lambda x: outcome_map.get(str(x).strip().lower(), 1)).values.astype(np.int64)
    else:
        y_outcome = None

    # flag_state (0..3): blue_flag_captured + 2 * red_flag_captured
    # derived from indices 10 and 11 of global state (the first 19 features of context)
    blue_flag_cap = X_raw[:, 10] > 0.5
    red_flag_cap = X_raw[:, 11] > 0.5
    y_flag = (blue_flag_cap.astype(np.int64) + 2 * red_flag_cap.astype(np.int64))

    targets = {
        "opponent_id": (y_opp, 7 if y_opp is not None else 0),
        "phase_id": (y_phase, 6 if y_phase is not None else 0),
        "score_outcome": (y_outcome, 3 if y_outcome is not None else 0),
        "flag_state": (y_flag, 4),
    }

    # Normalize features
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - mean) / std

    print("\n--- Training Diagnostic Probes ---")
    results = {}

    for name, (y, num_classes) in targets.items():
        if y is None:
            print(f"Skipping {name} (column not present in CSV).")
            continue

        # Filter invalid target values (e.g. -1 for opponent_id)
        valid = (y >= 0) & (y < num_classes)
        if not np.any(valid):
            print(f"Skipping {name} (no valid labels).")
            continue

        X_valid = X_norm[valid]
        y_valid = y[valid]

        # Train/Test split
        n_samples = len(X_valid)
        split = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, y_train = torch.tensor(X_valid[train_idx]), torch.tensor(y_valid[train_idx])
        X_test, y_test = torch.tensor(X_valid[test_idx]), torch.tensor(y_valid[test_idx])

        # Baseline: majority class accuracy
        unique, counts = np.unique(y_valid[test_idx], return_counts=True)
        majority_acc = (counts.max() / len(test_idx)) * 100.0 if len(test_idx) > 0 else 0.0

        # Build simple MLP classifier
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Linear(95, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Train loop
        model.train()
        for epoch in range(args.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Test evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            test_acc = (preds == y_test.numpy()).mean() * 100.0

        print(f"Target: {name:<15} | Samples: {n_samples:<6} | Majority-Class Baseline: {majority_acc:6.2f}% | Probe Accuracy: {test_acc:6.2f}%")
        results[name] = (test_acc, majority_acc)

    print("\n--- Diagnostic Conclusion ---")
    passes = []
    for name, (acc, baseline) in results.items():
        if acc > baseline + 5.0: # accuracy is significantly better than baseline
            passes.append(name)

    if len(passes) >= 3:
        print("RESULT: PASS")
        print(f"The 95-d temporal context contains strong predictive signal for: {', '.join(passes)}.")
        print("posterior/representation collapse is due to training / actor utilization, NOT lack of context signal.")
    else:
        print("RESULT: FAIL")
        print("The 95-d temporal context lacks predictive signal (not significantly better than majority class baseline).")
        print("Consider adding better global-state features (team shape, pressure, carrier information).")

if __name__ == "__main__":
    main()

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
    ctx_cols = [col for col in df.columns if str(col).startswith("q_phi_context_")]
    num_ctx = len(ctx_cols)
    if num_ctx == 0:
        print("Error: No context columns found in CSV.")
        sys.exit(1)

    # 1. Extract context features
    X_raw = df[ctx_cols].values.astype(np.float32)

    # Find qphi_logits columns
    logit_cols = [col for col in df.columns if str(col).startswith("qlogit_")]
    if len(logit_cols) > 0:
        X_logits = df[logit_cols].values.astype(np.float32)
    else:
        X_logits = None

    # 2. Extract targets
    # opponent_id (integer)
    if "opponent_id" in df.columns:
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
        "flag_state": (y_flag, 4),
        "score_outcome": (y_outcome, 3 if y_outcome is not None else 0),
    }

    # Normalize features
    mean_ctx = X_raw.mean(axis=0)
    std_ctx = X_raw.std(axis=0) + 1e-8
    X_ctx_norm = (X_raw - mean_ctx) / std_ctx

    if X_logits is not None:
        mean_logits = X_logits.mean(axis=0)
        std_logits = X_logits.std(axis=0) + 1e-8
        X_logits_norm = (X_logits - mean_logits) / std_logits
    else:
        X_logits_norm = None

    results = {}

    def run_probe(X_norm, y, num_classes):
        # Filter invalid target values (e.g. -1 for opponent_id)
        valid = (y >= 0) & (y < num_classes)
        if not np.any(valid):
            return None

        X_valid = X_norm[valid]
        y_valid = y[valid]

        # Seed for reproducibility of training and split
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = len(X_valid)
        split = int(n_samples * 0.8)
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:split], indices[split:]

        if len(test_idx) == 0 or len(train_idx) == 0:
            return None

        X_train, y_train = torch.tensor(X_valid[train_idx]), torch.tensor(y_valid[train_idx])
        X_test, y_test = torch.tensor(X_valid[test_idx]), torch.tensor(y_valid[test_idx])

        # Baselines
        unique_test, counts_test = np.unique(y_valid[test_idx], return_counts=True)
        majority_acc = (counts_test.max() / len(test_idx)) * 100.0
        random_acc = (1.0 / num_classes) * 100.0

        # Build simple MLP classifier
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_valid.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
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
            y_test_np = y_test.numpy()
            
            acc = (preds == y_test_np).mean() * 100.0
            
            # Balanced accuracy
            classes_present = np.unique(y_test_np)
            recalls = []
            for c in classes_present:
                mask = (y_test_np == c)
                if np.sum(mask) > 0:
                    recall = np.sum((preds == c) & mask) / np.sum(mask)
                    recalls.append(recall)
            bal_acc = np.mean(recalls) * 100.0 if recalls else 0.0

        return {
            "acc": acc,
            "bal_acc": bal_acc,
            "majority_acc": majority_acc,
            "random_acc": random_acc,
            "samples": n_samples
        }

    # Run probes
    for target_name, (y, num_classes) in targets.items():
        if y is None:
            continue
        
        # Probe context
        res_ctx = run_probe(X_ctx_norm, y, num_classes)
        if res_ctx is not None:
            results[f"context_{target_name}"] = res_ctx

        # Probe logits
        if X_logits_norm is not None:
            res_logits = run_probe(X_logits_norm, y, num_classes)
            if res_logits is not None:
                results[f"qphi_logits_{target_name}"] = res_logits

    # Print the specific format requested
    print("\n--- Diagnostic Probe Outputs ---")

    # 1. Context accuracy for each target
    targets_ordered = [("opponent_id", "opponent_acc"), ("phase_id", "phase_acc"), ("flag_state", "flag_acc"), ("score_outcome", "outcome_acc")]
    for t_id, print_name in targets_ordered:
        key = f"context_{t_id}"
        val_str = f"{results[key]['acc']:.2f}%" if key in results else "N/A"
        print(f"[probe/context] {print_name}: {val_str}")
    print()

    # 2. Qphi logits accuracy for each target
    for t_id, print_name in targets_ordered:
        key = f"qphi_logits_{t_id}"
        val_str = f"{results[key]['acc']:.2f}%" if key in results else "N/A"
        print(f"[probe/qphi_logits] {print_name}: {val_str}")
    print()

    # 3. Balanced accuracy for opponent target
    key_opp_ctx = "context_opponent_id"
    val_opp_ctx = f"{results[key_opp_ctx]['bal_acc']:.2f}%" if key_opp_ctx in results else "N/A"
    print(f"[probe/context] opponent_bal_acc: {val_opp_ctx}")

    key_opp_log = "qphi_logits_opponent_id"
    val_opp_log = f"{results[key_opp_log]['bal_acc']:.2f}%" if key_opp_log in results else "N/A"
    print(f"[probe/qphi_logits] opponent_bal_acc: {val_opp_log}")
    print()

    # 4. Baselines for each target
    for t_id, _ in targets_ordered:
        short_name = t_id.replace("_id", "").replace("_state", "").replace("state", "").replace("score_", "")
        key = f"context_{t_id}"
        if key in results:
            print(f"[probe] random_baseline_acc for {short_name}: {results[key]['random_acc']:.2f}%")
            print(f"[probe] majority_baseline_acc for {short_name}: {results[key]['majority_acc']:.2f}%")
        else:
            print(f"[probe] random_baseline_acc for {short_name}: N/A")
            print(f"[probe] majority_baseline_acc for {short_name}: N/A")

if __name__ == "__main__":
    main()

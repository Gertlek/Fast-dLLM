import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# Approx LAB cluster centers for common textile colors
COLOR_CENTERS: Dict[str, Tuple[float, float, float]] = {
    "red":    (53.0, 80.0, 67.0),
    "green":  (46.0, -51.0, 49.0),
    "blue":   (32.0, 79.0, -108.0),
    "yellow": (97.0, -21.0, 94.0),
    "black":  (8.0, 0.0, 0.0),
    "white":  (96.0, 0.0, 2.0),
    "denim":  (55.0, -2.0, -35.0),  # desaturated blue-ish
}

# Cluster covariances to generate realistic shapes with correlation
COVS: Dict[str, np.ndarray] = {
    "red": np.array([
        [45.0, 8.0, 12.0],
        [8.0, 60.0, 15.0],
        [12.0, 15.0, 55.0]
    ]),
    "green": np.array([
        [50.0, -6.0, 10.0],
        [-6.0, 65.0, 12.0],
        [10.0, 12.0, 58.0]
    ]),
    "blue": np.array([
        [48.0, 6.0, -10.0],
        [6.0, 58.0, -15.0],
        [-10.0, -15.0, 70.0]
    ]),
    "yellow": np.array([
        [42.0, -3.0, 8.0],
        [-3.0, 55.0, 10.0],
        [8.0, 10.0, 52.0]
    ]),
    "black": np.array([
        [38.0, 3.0, 4.0],
        [3.0, 50.0, 6.0],
        [4.0, 6.0, 48.0]
    ]),
    "white": np.array([
        [40.0, 2.0, -3.0],
        [2.0, 52.0, 5.0],
        [-3.0, 5.0, 46.0]
    ]),
    "denim": np.array([
        [52.0, 5.0, -12.0],
        [5.0, 62.0, -10.0],
        [-12.0, -10.0, 68.0]
    ]),
}


def make_dataset(n_per_color: int,
                 mislabel_rate: float,
                 crossmix_rate: float,
                 seed: int,
                 n_unknown: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    colors = list(COLOR_CENTERS.keys())
    rows = []
    pid = 1

    for color in colors:
        n = n_per_color
        n_cross = int(n * crossmix_rate)
        n_main = n - n_cross

        # Main samples from its own center
        pts_main = rng.multivariate_normal(COLOR_CENTERS[color], COVS[color], size=n_main)
        for p in pts_main:
            assigned = color
            if rng.random() < mislabel_rate:
                assigned = rng.choice([c for c in colors if c != assigned])
            rows.append({
                "id": pid,
                "L": p[0], "a": p[1], "b": p[2],
                "true_color": color,
                "assigned_bin": assigned,
                "source_center": color,
                "note": "own-cluster"
            })
            pid += 1

        # Cross-mixed samples drawn from other centers but assigned to this bin
        other_colors = [c for c in colors if c != color]
        if n_cross > 0:
            mix_sources = rng.choice(other_colors, size=n_cross, replace=True)
            for src in mix_sources:
                p = rng.multivariate_normal(COLOR_CENTERS[src], COVS[src], size=1)[0]
                assigned = color
                if rng.random() < mislabel_rate:
                    assigned = rng.choice([c for c in colors if c != assigned])
                rows.append({
                    "id": pid,
                    "L": p[0], "a": p[1], "b": p[2],
                    "true_color": src,
                    "assigned_bin": assigned,
                    "source_center": src,
                    "note": f"cross-from-{src}"
                })
                pid += 1

    # Add unknown samples - samples far from all known clusters
    if n_unknown > 0:
        for _ in range(n_unknown):
            # Generate points in regions far from known clusters
            # Strategy: pick random positions in LAB space avoiding known centers
            L = rng.uniform(0, 100)
            a = rng.uniform(-128, 127)
            b = rng.uniform(-128, 127)
            
            # Check distance to all known centers and regenerate if too close
            point = np.array([L, a, b])
            min_dist = min(np.linalg.norm(point - np.array(center)) 
                          for center in COLOR_CENTERS.values())
            
            # If too close to a known center, push it further away
            if min_dist < 50:
                # Find furthest corner from all centers
                furthest_corner = None
                max_min_dist = 0
                for trial_L in [10, 50, 90]:
                    for trial_a in [-100, 0, 100]:
                        for trial_b in [-100, 0, 100]:
                            trial_point = np.array([trial_L, trial_a, trial_b])
                            trial_min_dist = min(np.linalg.norm(trial_point - np.array(center)) 
                                               for center in COLOR_CENTERS.values())
                            if trial_min_dist > max_min_dist:
                                max_min_dist = trial_min_dist
                                furthest_corner = trial_point
                
                # Add noise around furthest corner
                L = furthest_corner[0] + rng.normal(0, 10)
                a = furthest_corner[1] + rng.normal(0, 20)
                b = furthest_corner[2] + rng.normal(0, 20)
            
            rows.append({
                "id": pid,
                "L": L,
                "a": a,
                "b": b,
                "true_color": "unknown",
                "assigned_bin": "unknown",
                "source_center": "unknown",
                "note": "outlier"
            })
            pid += 1

    df = pd.DataFrame(rows)
    df["L"] = df["L"].clip(0, 100)
    df["a"] = df["a"].clip(-128, 127)
    df["b"] = df["b"].clip(-128, 127)
    return df


@dataclass
class Radius2Classifier:
    X: np.ndarray
    y: np.ndarray
    radius: float = 2.0
    max_distance: float = None  # Threshold for unknown detection

    def predict_one(self, x: np.ndarray) -> Tuple[str, bool]:
        """Predict based on fixed radius balls around training points."""
        dists = np.linalg.norm(self.X - x, axis=1)
        inball = dists <= self.radius
        min_dist = np.min(dists)
        
        # Check if too far from all training points
        if self.max_distance is not None and min_dist > self.max_distance:
            return "unknown", False
        
        if inball.any():
            # If inside one or more balls, return label of closest point
            idx = np.argmin(dists + (~inball) * 1e9)
            return self.y[idx], True
        else:
            # If outside all balls, return label of nearest training point
            idx = np.argmin(dists)
            return self.y[idx], False

    def predict(self, X: np.ndarray) -> Tuple[List[str], List[bool]]:
        results = [self.predict_one(x) for x in X]
        predictions = [r[0] for r in results]
        coverages = [r[1] for r in results]
        return predictions, coverages
    
    def predict_with_coverage(self, X: np.ndarray) -> Tuple[List[str], float]:
        """Predict and return coverage (fraction of points inside any ball)."""
        predictions, coverages = self.predict(X)
        coverage = sum(coverages) / len(X) if len(X) > 0 else 0.0
        return predictions, coverage


@dataclass
class KNNClassifier:
    X: np.ndarray
    y: np.ndarray
    k: int = 5
    max_distance: float = None  # Threshold for unknown detection

    def predict_one(self, x: np.ndarray) -> Tuple[str, float]:
        """Predict based on k nearest neighbors with confidence."""
        dists = np.linalg.norm(self.X - x, axis=1)
        k_indices = np.argpartition(dists, min(self.k, len(dists) - 1))[:self.k]
        k_dists = dists[k_indices]
        k_labels = self.y[k_indices]
        
        # Check if nearest neighbors are too far
        avg_dist = np.mean(k_dists)
        if self.max_distance is not None and avg_dist > self.max_distance:
            return "unknown", 0.0
        
        # Majority vote with confidence
        unique, counts = np.unique(k_labels, return_counts=True)
        winner_idx = np.argmax(counts)
        confidence = counts[winner_idx] / self.k
        
        return unique[winner_idx], confidence

    def predict(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        results = [self.predict_one(x) for x in X]
        predictions = [r[0] for r in results]
        confidences = [r[1] for r in results]
        return predictions, confidences
    

def main():
    ap = argparse.ArgumentParser(description="Simulate LAB textile colors and evaluate a radius-2 rule.")
    ap.add_argument("--outdir", default=".", type=str, help="Output directory")
    ap.add_argument("--n-per-color", default=300, type=int, help="Samples per color in each split")
    ap.add_argument("--n-unknown", default=50, type=int, help="Number of unknown samples in test set")
    ap.add_argument("--mislabel-rate", default=0.08, type=float, help="Fraction of assigned_bin flips")
    ap.add_argument("--crossmix-rate", default=0.12, type=float, help="Fraction drawn from other centers")
    ap.add_argument("--radius", default=7.5, type=float, help="Radius for the rule")
    ap.add_argument("--radius-max-distance", default=30.0, type=float, help="Max distance for radius unknown detection")
    ap.add_argument("--k", default=5, type=int, help="K for KNN classifier")
    ap.add_argument("--knn-max-distance", default=30.0, type=float, help="Max distance for KNN unknown detection")
    ap.add_argument("--seed", default=123, type=int, help="Base RNG seed")
    ap.add_argument("--plot", action="store_true", help="3D scatter of a sample if matplotlib is available")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Training set with mislabels to reflect sorter noise (no unknowns in training)
    df_train = make_dataset(
        n_per_color=args.n_per_color,
        mislabel_rate=args.mislabel_rate,
        crossmix_rate=args.crossmix_rate,
        seed=args.seed,
        n_unknown=0  # No unknowns in training
    )

    # Test set without mislabels but with overlap and unknown samples
    df_test = make_dataset(
        n_per_color=args.n_per_color,
        mislabel_rate=args.mislabel_rate,
        crossmix_rate=args.crossmix_rate,
        seed=args.seed + 500,
        n_unknown=args.n_unknown
    ).rename(columns={"assigned_bin": "intended_bin"})

    train_csv = os.path.join(args.outdir, "textile_lab_train.csv")
    test_csv = os.path.join(args.outdir, "textile_lab_test.csv")
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Fit rule on assigned_bin
    X_train = df_train[["L", "a", "b"]].to_numpy()
    y_train = df_train["assigned_bin"].to_numpy()
    clf = Radius2Classifier(X_train, y_train, radius=args.radius, max_distance=args.radius_max_distance)

    # Evaluate against true_color
    X_test = df_test[["L", "a", "b"]].to_numpy()
    y_true = df_test["true_color"].to_numpy()
    y_pred, coverage = clf.predict_with_coverage(X_test)
    y_pred = np.array(y_pred)
    acc = float((y_pred == y_true).mean())

    # KNN classifier with unknown detection
    knn_clf = KNNClassifier(X_train, y_train, k=args.k, max_distance=args.knn_max_distance)
    y_pred_knn, confidences = knn_clf.predict(X_test)
    y_pred_knn = np.array(y_pred_knn)
    acc_knn = float((y_pred_knn == y_true).mean())

    # Per class accuracy (including unknown)
    classes = sorted(list(COLOR_CENTERS.keys()) + ["unknown"])
    per_class = {}
    per_class_coverage = {}
    per_class_knn = {}
    
    for c in classes:
        mask = y_true == c
        if mask.any():
            per_class[c] = float((y_pred[mask] == y_true[mask]).mean())
            per_class_knn[c] = float((y_pred_knn[mask] == y_true[mask]).mean())
            # Calculate coverage for this class
            class_X = X_test[mask]
            class_covered = 0
            for x in class_X:
                dists = np.linalg.norm(clf.X - x, axis=1)
                if (dists <= clf.radius).any():
                    class_covered += 1
            per_class_coverage[c] = class_covered / len(class_X)
        else:
            per_class[c] = float("nan")
            per_class_knn[c] = float("nan")
            per_class_coverage[c] = float("nan")

    # Additional metrics for unknown detection
    unknown_mask = y_true == "unknown"
    known_mask = y_true != "unknown"
    
    radius_unknown_detection = {
        "radius_unknown_recall": float((y_pred[unknown_mask] == "unknown").mean()) if unknown_mask.any() else 0.0,
        "radius_unknown_precision": float((y_true[y_pred == "unknown"] == "unknown").mean()) if (y_pred == "unknown").any() else 0.0,
        "radius_known_accuracy": float((y_pred[known_mask] == y_true[known_mask]).mean()) if known_mask.any() else 0.0,
    }
    
    knn_unknown_detection = {
        "knn_unknown_recall": float((y_pred_knn[unknown_mask] == "unknown").mean()) if unknown_mask.any() else 0.0,
        "knn_unknown_precision": float((y_true[y_pred_knn == "unknown"] == "unknown").mean()) if (y_pred_knn == "unknown").any() else 0.0,
        "knn_known_accuracy": float((y_pred_knn[known_mask] == y_true[known_mask]).mean()) if known_mask.any() else 0.0,
        "avg_confidence_correct": float(np.mean([confidences[i] for i in range(len(confidences)) if y_pred_knn[i] == y_true[i]])),
        "avg_confidence_incorrect": float(np.mean([confidences[i] for i in range(len(confidences)) if y_pred_knn[i] != y_true[i]])) if any(y_pred_knn[i] != y_true[i] for i in range(len(y_pred_knn))) else 0.0,
    }

    metrics = {
        "radius": args.radius,
        "radius_max_distance": args.radius_max_distance,
        "k": args.k,
        "knn_max_distance": args.knn_max_distance,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_unknown_test": int(unknown_mask.sum()),
        "radius_accuracy_overall": acc,
        "knn_accuracy_overall": acc_knn,
        "coverage_overall": coverage,
        "radius_accuracy_per_class": per_class,
        "knn_accuracy_per_class": per_class_knn,
        "coverage_per_class": per_class_coverage,
        "radius_unknown_detection": radius_unknown_detection,
        "knn_unknown_detection": knn_unknown_detection,
        "train_csv": train_csv,
        "test_csv": test_csv,
    }

    metrics_json = os.path.join(args.outdir, "radius2_metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a small model blob for reuse
    model_blob = {
        "radius": args.radius,
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "labels": classes,
        "color_centers": COLOR_CENTERS,
    }
    model_json = os.path.join(args.outdir, "radius2_model.json")
    with open(model_json, "w") as f:
        json.dump(model_blob, f)

    print(json.dumps(metrics, indent=2))

    # Always plot if matplotlib is available
    if HAVE_MPL:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            
            # Color mapping for visualization
            color_map = {
                "red": "#FF0000",
                "green": "#00FF00",
                "blue": "#0000FF",
                "yellow": "#FFFF00",
                "black": "#000000",
                "white": "#FFFFFF",
                "denim": "#00FFFF"
            }
            
            # Plot 1: Training points only
            sample_plot = df_train.sample(min(1500, len(df_train)), random_state=0)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            
            # Plot each color separately with its own color
            for color_name in COLOR_CENTERS.keys():
                mask = sample_plot["source_center"] == color_name
                subset = sample_plot[mask]
                ax.scatter(
                    subset["L"], 
                    subset["a"], 
                    subset["b"], 
                    c=color_map.get(color_name, "gray"),
                    label=color_name,
                    s=20,
                    alpha=0.6,
                    edgecolors='none'
                )
            
            ax.set_xlabel("L (Lightness)", fontsize=10)
            ax.set_ylabel("a (Green-Red)", fontsize=10)
            ax.set_zlabel("b (Blue-Yellow)", fontsize=10)
            ax.set_title("Training Set in LAB Color Space", fontsize=12)
            ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "lab_train_scatter.png"), dpi=150)
            print(f"Training plot saved to {os.path.join(args.outdir, 'lab_train_scatter.png')}")
            plt.close()
            
            # Plot 1b: Test set with unknowns
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            
            # Plot each known color from test set
            for color_name in COLOR_CENTERS.keys():
                mask = df_test["true_color"] == color_name
                subset = df_test[mask]
                ax.scatter(
                    subset["L"], 
                    subset["a"], 
                    subset["b"], 
                    c=color_map.get(color_name, "gray"),
                    label=color_name,
                    s=20,
                    alpha=0.6,
                    edgecolors='none'
                )
            
            # Plot unknown samples from test set
            unknown_test = df_test[df_test["true_color"] == "unknown"]
            if len(unknown_test) > 0:
                ax.scatter(
                    unknown_test["L"], 
                    unknown_test["a"], 
                    unknown_test["b"], 
                    c='gray',
                    marker='*',
                    s=150,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=1.5,
                    label="unknown"
                )
            
            ax.set_xlabel("L (Lightness)", fontsize=10)
            ax.set_ylabel("a (Green-Red)", fontsize=10)
            ax.set_zlabel("b (Blue-Yellow)", fontsize=10)
            ax.set_title("Test Set in LAB Color Space (with unknowns)", fontsize=12)
            ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "lab_test_scatter.png"), dpi=150)
            print(f"Test plot saved to {os.path.join(args.outdir, 'lab_test_scatter.png')}")
            plt.close()
            
            # Plot 2: Misclassified test points (RADIUS) with radius balls
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            
            # Draw semi-transparent radius balls around training points
            sample_indices = np.random.choice(len(X_train), size=min(80, len(X_train)), replace=False)
            
            # Draw spheres around sampled training points with better visibility
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x_sphere = args.radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = args.radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = args.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            for idx in sample_indices:
                center = X_train[idx]
                ax.plot_surface(
                    x_sphere + center[0],
                    y_sphere + center[1],
                    z_sphere + center[2],
                    alpha=0.08,
                    color='cyan',
                    edgecolor='none'
                )
            
            # Plot training points (small dots at sphere centers)
            ax.scatter(X_train[sample_indices, 0], X_train[sample_indices, 1], X_train[sample_indices, 2],
                      c='blue', s=5, alpha=0.3, label='training centers')
            
            # Separate correct and incorrect predictions
            correct_mask = y_pred == y_true
            incorrect_mask = ~correct_mask
            
            # Plot correctly classified (small, faded)
            correct_df = df_test[correct_mask].copy()
            for color_name in classes:
                if color_name == "unknown":
                    continue
                mask = correct_df["true_color"] == color_name
                subset = correct_df[mask]
                if len(subset) > 0:
                    ax.scatter(
                        subset["L"], subset["a"], subset["b"],
                        c=color_map.get(color_name, "gray"),
                        s=15, alpha=0.25, edgecolors='none'
                    )
            
            # Plot ONLY misclassified points (large, bold)
            incorrect_df = df_test[incorrect_mask].copy()
            incorrect_df["predicted"] = y_pred[incorrect_mask]
            
            for color_name in classes:
                mask = incorrect_df["true_color"] == color_name
                subset = incorrect_df[mask]
                if len(subset) > 0:
                    if color_name == "unknown":
                        ax.scatter(
                            subset["L"], subset["a"], subset["b"],
                            c='gray', marker='*', s=200, alpha=0.95,
                            edgecolors='red', linewidths=2,
                            label=f"unknown (missed)"
                        )
                    else:
                        ax.scatter(
                            subset["L"], subset["a"], subset["b"],
                            c=color_map.get(color_name, "gray"),
                            s=120, alpha=0.95, edgecolors='black', linewidths=1.5
                        )
            
            ax.set_xlabel("L (Lightness)", fontsize=10)
            ax.set_ylabel("a (Green-Red)", fontsize=10)
            ax.set_zlabel("b (Blue-Yellow)", fontsize=10)
            ax.set_title(f"Radius (r={args.radius}): Misclassified (bold) vs Correct (faded)\nAccuracy: {acc:.2%}, Coverage: {coverage:.2%} | Unknown Recall: {radius_unknown_detection['radius_unknown_recall']:.2%}", fontsize=12)
            ax.legend(loc='upper left', fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "lab_misclassified.png"), dpi=150)
            print(f"Misclassification plot saved to {os.path.join(args.outdir, 'lab_misclassified.png')}")
            plt.close()
            
        except Exception as e:
            print(f"Plot failed: {e}")

if __name__ == "__main__":
    main()
import json
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load and parse JSON data from file."""
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
        if not content:
            return []
        if content.startswith("[") and content.endswith("]"):
            return json.loads(content)
        else:
            content = re.sub(r"}\s*{", "},{", content)
            return json.loads(f"[{content}]")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"CRITICAL ERROR loading {file_path}: {e}")
        return []


def simple_feature_extraction(logs):
    """Simple feature extraction for LOF analysis."""

    # Basic syscall weights for attack detection
    attack_indicators = {
        "ptrace": 10.0,
        "process_vm_readv": 10.0,
        "process_vm_writev": 10.0,
        "kexec_load": 8.0,
        "init_module": 8.0,
        "delete_module": 6.0,
        "capset": 5.0,
        "setuid": 4.0,
        "setgid": 4.0,
        "execve": 3.0,
        "clone": 2.0,
        "fork": 1.0,
        "socket": 2.0,
        "connect": 2.0,
        "mprotect": 3.0,
        "mmap": 2.0,
    }

    features = []

    for log in logs:
        syscalls = log.get("kernel", {}).get("syscall_counts", {})
        total = max(1, sum(syscalls.values()))

        # Basic attack score
        attack_score = 0
        for sc, cnt in syscalls.items():
            weight = attack_indicators.get(sc, 0.5)
            attack_score += cnt * weight

        normalized_attack_score = attack_score / total

        # Simple pattern features
        has_ptrace = 1 if "ptrace" in syscalls else 0
        has_process_vm = (
            1
            if any(sc in syscalls for sc in ["process_vm_readv", "process_vm_writev"])
            else 0
        )
        has_execve = 1 if "execve" in syscalls else 0
        has_setuid = 1 if "setuid" in syscalls else 0

        # Basic intensities
        critical_intensity = (
            sum(
                syscalls.get(sc, 0)
                for sc in [
                    "ptrace",
                    "process_vm_readv",
                    "process_vm_writev",
                    "kexec_load",
                ]
            )
            / total
        )
        execution_intensity = (
            sum(syscalls.get(sc, 0) for sc in ["execve", "clone", "fork"]) / total
        )
        network_intensity = (
            sum(syscalls.get(sc, 0) for sc in ["socket", "connect", "bind"]) / total
        )

        # Rare syscalls
        rare_syscalls = [sc for sc in syscalls if sc.startswith("syscall_")]
        rare_ratio = sum(syscalls.get(sc, 0) for sc in rare_syscalls) / total

        # Basic features
        unique_syscalls = len(syscalls)
        total_activity = np.log1p(total)

        features.append(
            [
                normalized_attack_score,
                has_ptrace,
                has_process_vm,
                has_execve,
                has_setuid,
                critical_intensity,
                execution_intensity,
                network_intensity,
                rare_ratio,
                unique_syscalls,
                total_activity,
            ]
        )

    columns = [
        "attack_score",
        "has_ptrace",
        "has_process_vm",
        "has_execve",
        "has_setuid",
        "critical_intensity",
        "execution_intensity",
        "network_intensity",
        "rare_ratio",
        "unique_syscalls",
        "total_activity",
    ]

    return pd.DataFrame(features, columns=columns)


def evaluate_lof_results(y_true, lof_scores):
    """Evaluate LOF results with basic metrics."""

    # LOF returns negative scores for outliers (lower = more anomalous)
    # We'll convert to positive scores where higher = more anomalous
    anomaly_scores = -lof_scores

    y_true = np.array(y_true)
    anomaly_scores = np.array(anomaly_scores)

    # Calculate percentiles for threshold selection
    attack_scores = anomaly_scores[y_true == 1]
    normal_scores = anomaly_scores[y_true == 0]

    print(f"\n📊 LOF Score Analysis:")
    print(
        f"  • Attack scores: min={attack_scores.min():.4f}, max={attack_scores.max():.4f}, mean={attack_scores.mean():.4f}"
    )
    print(
        f"  • Normal scores: min={normal_scores.min():.4f}, max={normal_scores.max():.4f}, mean={normal_scores.mean():.4f}"
    )
    print(
        f"  • Score separation: {np.mean(attack_scores) - np.mean(normal_scores):.4f}"
    )

    # Simple threshold-based evaluation
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n🔍 LOF Threshold Analysis:")
    print(
        f"{'Threshold':<10} {'Detection Rate':<15} {'False Positive Rate':<20} {'Precision':<10}"
    )
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        # Use percentile-based threshold
        score_threshold = np.percentile(anomaly_scores, threshold * 100)

        # Classify
        predictions = (anomaly_scores >= score_threshold).astype(int)

        # Calculate metrics
        tp = np.sum((y_true == 1) & (predictions == 1))
        fp = np.sum((y_true == 0) & (predictions == 1))
        tn = np.sum((y_true == 0) & (predictions == 0))
        fn = np.sum((y_true == 1) & (predictions == 0))

        detection_rate = tp / max(1, tp + fn)
        false_positive_rate = fp / max(1, fp + tn)
        precision = tp / max(1, tp + fp)
        f1_score = (
            2 * (precision * detection_rate) / max(0.001, precision + detection_rate)
        )

        print(
            f"{threshold:<10} {detection_rate:<15.3f} {false_positive_rate:<20.3f} {precision:<10.3f}"
        )

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
            best_metrics = {
                "threshold": score_threshold,
                "detection_rate": detection_rate,
                "false_positive_rate": false_positive_rate,
                "precision": precision,
                "f1_score": f1_score,
            }
            best_predictions = predictions

    # Detailed classification breakdown
    if best_predictions is not None:
        print(f"\n" + "=" * 60)
        print("DETAILED CLASSIFICATION BREAKDOWN")
        print("=" * 60)

        # Calculate detailed metrics
        attacks_as_attack = np.sum((y_true == 1) & (best_predictions == 1))
        attacks_as_safe = np.sum((y_true == 1) & (best_predictions == 0))

        normal_as_attack = np.sum((y_true == 0) & (best_predictions == 1))
        normal_as_safe = np.sum((y_true == 0) & (best_predictions == 0))

        total_attacks = np.sum(y_true == 1)
        total_normal = np.sum(y_true == 0)

        print(f"\n🎯 ATTACK LOGS ({total_attacks} total):")
        print(
            f"  ✓ Flagged as ATTACK:     {attacks_as_attack:4d} ({(attacks_as_attack/total_attacks):.1%})"
        )
        print(
            f"  ✗ Missed (flagged SAFE): {attacks_as_safe:4d} ({(attacks_as_safe/total_attacks):.1%})"
        )

        print(f"\n🛡️ NORMAL LOGS ({total_normal} total):")
        print(
            f"  ✗ False ATTACK flags:    {normal_as_attack:4d} ({(normal_as_attack/total_normal):.1%})"
        )
        print(
            f"  ✓ Correctly SAFE:        {normal_as_safe:4d} ({(normal_as_safe/total_normal):.1%})"
        )

        # Performance metrics
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  • Attack Detection Rate: {best_metrics['detection_rate']:.1%}")
        print(f"  • Critical Miss Rate: {(attacks_as_safe/total_attacks):.1%}")
        print(f"  • False Positive Rate: {(normal_as_attack/total_normal):.1%}")
        print(f"  • Precision: {best_metrics['precision']:.1%}")
        print(f"  • F1 Score: {best_metrics['f1_score']:.3f}")

    return best_metrics, anomaly_scores


def run_simple_lof_analysis():
    """Run simple LOF analysis on the dataset."""

    # File paths
    train_file = "ADFA_log/training_data_kernel_activity.json"
    attack_file = "ADFA_log/Attack_logs_json/all_attacks.json"
    validation_file = "ADFA_log/normal_validation.json"

    print("=" * 60)
    print("SIMPLE LOF ANOMALY DETECTION")
    print("=" * 60)

    print("\n[1/4] Loading datasets...")
    train_data = load_data(train_file)
    attack_data = load_data(attack_file)
    normal_validation_data = load_data(validation_file)

    print(f"  • Training samples: {len(train_data)}")
    print(f"  • Attack samples: {len(attack_data)}")
    print(f"  • Normal validation samples: {len(normal_validation_data)}")

    # Check for empty datasets
    if (
        len(train_data) == 0
        or len(attack_data) == 0
        or len(normal_validation_data) == 0
    ):
        print("  ❌ ERROR: One or more datasets are empty!")
        return None

    print("\n[2/4] Extracting simple features...")
    X_train_df = simple_feature_extraction(train_data)
    X_attack_df = simple_feature_extraction(attack_data)
    X_normal_val_df = simple_feature_extraction(normal_validation_data)

    print(f"  • Feature dimensions: {X_train_df.shape[1]}")
    print(f"  • Features: {', '.join(X_train_df.columns)}")

    print("\n[3/4] Training LOF model...")
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)

    # Train LOF on normal data only
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.1,
        novelty=False,  # Use fit_predict for training data
        metric="minkowski",
        p=2,
    )

    # Fit and predict on training data
    lof_scores_train = lof.fit_predict(X_train_scaled)

    print(f"  • LOF model trained with {lof.n_neighbors_} neighbors")
    print(f"  • Contamination estimate: {lof.contamination}")

    print("\n[4/4] Evaluating on validation data...")
    # Prepare combined validation data
    X_combined_df = pd.concat([X_attack_df, X_normal_val_df], ignore_index=True)
    y_true = np.array([1] * len(X_attack_df) + [0] * len(X_normal_val_df))

    # Scale validation data
    X_combined_scaled = scaler.transform(X_combined_df)

    # For novelty=False, we need to fit on the combined data to get scores
    # We'll use the negative_outlier_factor_ from the training data as reference
    lof_combined = LocalOutlierFactor(
        n_neighbors=20, contamination=0.1, novelty=False, metric="minkowski", p=2
    )

    # Fit on combined data to get outlier factors
    lof_combined.fit(X_combined_scaled)
    lof_scores_val = lof_combined.negative_outlier_factor_

    # Evaluate results
    best_metrics, anomaly_scores = evaluate_lof_results(y_true, lof_scores_val)

    if best_metrics:
        print(f"\n" + "=" * 60)
        print("SIMPLE LOF RESULTS SUMMARY")
        print("=" * 60)

        print(f"\n🎯 Best Performance (Threshold: {best_metrics['threshold']:.4f}):")
        print(f"  • Detection Rate: {best_metrics['detection_rate']:.1%}")
        print(f"  • False Positive Rate: {best_metrics['false_positive_rate']:.1%}")
        print(f"  • Precision: {best_metrics['precision']:.1%}")
        print(f"  • F1 Score: {best_metrics['f1_score']:.3f}")

        # Calculate additional metrics
        attack_scores = anomaly_scores[y_true == 1]
        normal_scores = anomaly_scores[y_true == 0]

        print(f"\n📊 Score Distribution:")
        print(
            f"  • Attack mean: {np.mean(attack_scores):.4f} (std: {np.std(attack_scores):.4f})"
        )
        print(
            f"  • Normal mean: {np.mean(normal_scores):.4f} (std: {np.std(normal_scores):.4f})"
        )
        print(f"  • Separation: {np.mean(attack_scores) - np.mean(normal_scores):.4f}")

        if np.mean(attack_scores) > np.mean(normal_scores):
            print(f"  ✅ Good separation: Attacks score higher than normals")
        else:
            print(f"  ⚠️ Poor separation: Attacks score lower than normals")

    return best_metrics


if __name__ == "__main__":
    print("\n🚀 Starting Simple LOF Analysis...\n")

    metrics = run_simple_lof_analysis()

    if metrics is None:
        print("\n❌ Analysis failed due to data loading issues.")
        exit(1)

    print("\n" + "=" * 60)
    print("SIMPLE LOF ANALYSIS COMPLETE")
    print("=" * 60)

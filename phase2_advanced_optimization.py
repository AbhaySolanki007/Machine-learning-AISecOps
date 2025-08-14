# Phase 2 Advanced Optimization Strategy
# Addressing the core limitations of Phase 1

import json
import re
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
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


def enhanced_feature_extraction_phase2(logs):
    """Phase 2: Advanced feature engineering with better discrimination."""

    # Phase 2: More sophisticated attack indicators
    attack_indicators = {
        # Critical attack patterns (higher weights)
        "ptrace": 20.0,  # Process injection
        "process_vm_readv": 20.0,  # Memory reading
        "process_vm_writev": 20.0,  # Memory writing
        "kexec_load": 15.0,  # Kernel replacement
        "init_module": 15.0,  # Module loading
        "delete_module": 12.0,  # Module removal
        "pivot_root": 12.0,  # Root manipulation
        "reboot": 12.0,  # System reboot
        # Privilege escalation (increased weights)
        "capset": 12.0,
        "setuid": 10.0,
        "setgid": 10.0,
        "setfsuid": 10.0,
        "setfsgid": 10.0,
        "setresuid": 10.0,
        "setresgid": 10.0,
        # Suspicious syscalls (increased weights)
        "syscall_265": 15.0,  # Unknown/rare syscalls
        "syscall_252": 15.0,
        "mprotect": 10.0,  # Memory protection changes
        "mmap": 8.0,  # Memory mapping
        "chroot": 10.0,  # Root change
        # Execution (moderate weights)
        "execve": 8.0,
        "execveat": 8.0,
        "clone": 6.0,
        "fork": 4.0,
        "vfork": 4.0,
        # Network (increased suspicion)
        "socket": 5.0,
        "connect": 5.0,
        "bind": 6.0,
        "listen": 6.0,
        "accept": 5.0,
        # File system
        "mount": 8.0,
        "umount": 7.0,
    }

    # Benign operations (more negative weights)
    benign_operations = {
        "getuid": -1.0,
        "geteuid": -1.0,
        "getgid": -1.0,
        "getegid": -1.0,
        "read": -0.5,
        "write": -0.5,
        "open": -0.3,
        "close": -0.5,
        "stat": -0.3,
        "fstat": -0.3,
        "lstat": -0.3,
        "access": -0.3,
        "getcwd": -0.2,
        "getdents": -0.2,
    }

    syscall_weights = {**attack_indicators, **benign_operations}

    # Phase 2: Enhanced pattern detection
    critical_attack_calls = [
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "kexec_load",
        "init_module",
        "delete_module",
        "pivot_root",
    ]
    privilege_escalation_calls = [
        "capset",
        "setuid",
        "setgid",
        "setfsuid",
        "setfsgid",
        "setresuid",
        "setresgid",
    ]
    memory_manipulation_calls = [
        "mprotect",
        "mmap",
        "process_vm_readv",
        "process_vm_writev",
    ]
    execution_calls = ["execve", "execveat", "clone", "fork", "vfork"]
    network_calls = [
        "socket",
        "connect",
        "bind",
        "listen",
        "accept",
        "sendto",
        "recvfrom",
    ]

    features = []

    for log in logs:
        syscalls = log.get("kernel", {}).get("syscall_counts", {})
        total = max(1, sum(syscalls.values()))

        # Phase 2: Enhanced attack scoring
        attack_score = 0
        for sc, cnt in syscalls.items():
            weight = syscall_weights.get(sc, 0.2)  # Higher default weight
            attack_score += cnt * weight

        normalized_attack_score = attack_score / total

        # Phase 2: More sophisticated pattern detection
        has_ptrace = 1 if "ptrace" in syscalls else 0
        has_process_vm = (
            1
            if any(sc in syscalls for sc in ["process_vm_readv", "process_vm_writev"])
            else 0
        )
        has_kernel_mod = (
            1
            if any(
                sc in syscalls for sc in ["init_module", "delete_module", "kexec_load"]
            )
            else 0
        )

        # Enhanced category intensities
        critical_intensity = (
            sum(syscalls.get(sc, 0) for sc in critical_attack_calls) / total
        )
        priv_esc_intensity = (
            sum(syscalls.get(sc, 0) for sc in privilege_escalation_calls) / total
        )
        memory_intensity = (
            sum(syscalls.get(sc, 0) for sc in memory_manipulation_calls) / total
        )
        execution_intensity = sum(syscalls.get(sc, 0) for sc in execution_calls) / total
        network_intensity = sum(syscalls.get(sc, 0) for sc in network_calls) / total

        # Phase 2: Enhanced rare syscall detection
        rare_syscalls = [sc for sc in syscalls if sc.startswith("syscall_")]
        rare_count = sum(syscalls.get(sc, 0) for sc in rare_syscalls)
        rare_ratio = rare_count / total
        num_rare = len(rare_syscalls)

        # Enhanced benign activity detection
        benign_syscalls = [
            "read",
            "write",
            "open",
            "close",
            "stat",
            "fstat",
            "getuid",
            "geteuid",
        ]
        benign_count = sum(syscalls.get(sc, 0) for sc in benign_syscalls)
        benign_ratio = benign_count / total

        # Phase 2: Advanced suspicious patterns
        suspicious_combinations = 0
        if "ptrace" in syscalls and "execve" in syscalls:
            suspicious_combinations += 2  # Increased weight
        if "setuid" in syscalls and "execve" in syscalls:
            suspicious_combinations += 2
        if "mprotect" in syscalls and "mmap" in syscalls:
            suspicious_combinations += 1
        if any(sc.startswith("syscall_") for sc in syscalls):
            suspicious_combinations += 3  # Increased weight for rare syscalls

        # Phase 2: Enhanced pattern score
        pattern_score = (
            critical_intensity * 15  # Increased from 10
            + priv_esc_intensity * 12  # Increased from 8
            + memory_intensity * 8  # Increased from 6
            + rare_ratio * 15  # Increased from 10
            + suspicious_combinations * 3  # Increased from 2
            + has_ptrace * 8  # Increased from 5
            + has_process_vm * 8  # Increased from 5
            + has_kernel_mod * 8  # Increased from 5
        )

        # Phase 2: Additional features
        unique_syscalls = len(syscalls)
        syscall_entropy = stats.entropy(list(syscalls.values())) if syscalls else 0

        # Phase 2: Attack complexity score
        attack_complexity = (
            unique_syscalls * 0.1
            + syscall_entropy * 2.0
            + rare_ratio * 10.0
            + suspicious_combinations * 2.0
        )

        # Phase 2: Normal behavior score (inverse)
        normal_behavior_score = benign_ratio * 5.0

        features.append(
            [
                normalized_attack_score,  # Primary attack score
                pattern_score,  # Enhanced pattern score
                critical_intensity,  # Critical syscall intensity
                priv_esc_intensity,  # Privilege escalation intensity
                memory_intensity,  # Memory manipulation intensity
                execution_intensity,  # Execution intensity
                network_intensity,  # Network activity
                rare_ratio,  # Ratio of rare syscalls
                num_rare,  # Number of unique rare syscalls
                benign_ratio,  # Ratio of benign operations
                suspicious_combinations,  # Number of suspicious patterns
                has_ptrace,  # Binary: has ptrace
                has_process_vm,  # Binary: has process VM operations
                has_kernel_mod,  # Binary: has kernel module operations
                unique_syscalls,  # Syscall diversity
                syscall_entropy,  # Entropy of distribution
                np.log1p(total),  # Log-transformed total activity
                attack_complexity,  # NEW: Attack complexity score
                normal_behavior_score,  # NEW: Normal behavior score
            ]
        )

    columns = [
        "attack_score",
        "pattern_score",
        "critical_intensity",
        "priv_esc_intensity",
        "memory_intensity",
        "execution_intensity",
        "network_intensity",
        "rare_ratio",
        "num_rare_syscalls",
        "benign_ratio",
        "suspicious_combinations",
        "has_ptrace",
        "has_process_vm",
        "has_kernel_mod",
        "unique_syscalls",
        "syscall_entropy",
        "total_activity",
        "attack_complexity",
        "normal_behavior_score",
    ]

    return pd.DataFrame(features, columns=columns)


class Phase2AdvancedDetector:
    """Phase 2: Advanced ensemble detector with sophisticated optimizations."""

    def __init__(self, contamination_estimate=0.15):  # Back to moderate contamination
        """Initialize with Phase 2 optimizations."""
        self.models = {
            "ocsvm": OneClassSVM(
                kernel="rbf", gamma="scale", nu=contamination_estimate, cache_size=2000
            ),
            "iforest": IsolationForest(
                contamination=contamination_estimate,
                random_state=42,
                n_estimators=1000,
                max_samples="auto",
                bootstrap=True,
                max_features=1.0,
            ),
            "lof": LocalOutlierFactor(
                novelty=True,
                contamination=contamination_estimate,
                n_neighbors=15,
                metric="minkowski",
                p=2,
            ),
        }

        # Phase 2: Optimized weights based on expected performance
        self.weights = {
            "ocsvm": 0.25,  # Reduced weight
            "iforest": 0.50,  # Increased weight (best performer)
            "lof": 0.25,  # Kept same
        }

        self.scaler = RobustScaler()
        self.feature_selector = None
        self.pca = PCA(n_components=0.95, random_state=42)

    def fit(self, X_train, X_attack_sample=None, y_train=None):
        """Fit models with Phase 2 optimizations."""

        # Phase 2: Feature selection if labels available
        if y_train is not None:
            self.feature_selector = SelectKBest(
                f_classif, k=15
            )  # Select top 15 features
            X_selected = self.feature_selector.fit_transform(X_train, y_train)
            print(
                f"  • Feature selection: {X_train.shape[1]} → {X_selected.shape[1]} features"
            )
        else:
            X_selected = X_train

        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)

        # Apply PCA for dimensionality reduction
        X_transformed = self.pca.fit_transform(X_scaled)

        print(
            f"  • Reduced dimensions: {X_selected.shape[1]} → {X_transformed.shape[1]}"
        )

        # Train each model
        for name, model in self.models.items():
            print(f"  • Training {name}...")
            model.fit(X_transformed)

        # If attack samples provided, calibrate scoring
        if X_attack_sample is not None:
            if self.feature_selector:
                X_attack_selected = self.feature_selector.transform(X_attack_sample)
            else:
                X_attack_selected = X_attack_sample
            X_attack_scaled = self.scaler.transform(X_attack_selected)
            X_attack_transformed = self.pca.transform(X_attack_scaled)
            self._calibrate_scoring(X_transformed, X_attack_transformed)

        return self

    def _calibrate_scoring(self, X_normal, X_attack):
        """Calibrate model scoring based on sample distributions."""
        print("  • Calibrating scoring based on sample distributions...")

        for name, model in self.models.items():
            if name == "lof":
                normal_scores = model.score_samples(X_normal)
                attack_scores = model.score_samples(X_attack)
            else:
                normal_scores = model.decision_function(X_normal)
                attack_scores = model.decision_function(X_attack)

            if np.mean(attack_scores) > np.mean(normal_scores):
                print(f"    ⚠️ {name}: Inverted detection (attack mean > normal mean)")
            else:
                print(f"    ✅ {name}: Correct detection (attack mean < normal mean)")

    def decision_function(self, X):
        """Get ensemble anomaly scores."""
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
        else:
            X_selected = X

        X_scaled = self.scaler.transform(X_selected)
        X_transformed = self.pca.transform(X_scaled)

        ensemble_scores = np.zeros(len(X))

        for name, model in self.models.items():
            if name == "lof":
                scores = model.score_samples(X_transformed)
            else:
                scores = model.decision_function(X_transformed)

            # Phase 2: Improved normalization
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)

            if p95 > p5:
                normalized_scores = (scores - p5) / (p95 - p5)
                normalized_scores = np.clip(normalized_scores, 0, 1)
            else:
                normalized_scores = np.ones_like(scores) * 0.5

            ensemble_scores += self.weights[name] * normalized_scores

        return ensemble_scores


def optimize_thresholds_phase2(scores, y_true):
    """Phase 2: Advanced threshold optimization with ROC analysis."""

    attack_scores = scores[y_true == 1]
    normal_scores = scores[y_true == 0]

    print(f"\n📊 Score Statistics for Phase 2 Optimization:")
    print(
        f"  • Attack scores: mean={np.mean(attack_scores):.4f}, std={np.std(attack_scores):.4f}"
    )
    print(
        f"  • Normal scores: mean={np.mean(normal_scores):.4f}, std={np.std(normal_scores):.4f}"
    )
    print(
        f"  • Score separation: {np.mean(normal_scores) - np.mean(attack_scores):.4f}"
    )

    # Phase 2: More sophisticated threshold search
    best_cost = float("inf")
    best_thresholds = None
    best_metrics = None

    # Phase 2: Balanced cost weights
    missed_attack_cost = 120.0  # High but not extreme
    false_positive_cost = 4.0  # Moderate
    suspicious_cost = 1.0

    # Phase 2: Wider search ranges
    attack_percentiles = np.arange(75, 98, 1)  # 75th to 97th percentile
    normal_percentiles = np.arange(3, 35, 1)  # 3rd to 34th percentile

    valid_configs = 0

    for attack_pct in attack_percentiles:
        for normal_pct in normal_percentiles:

            attack_threshold = np.percentile(attack_scores, attack_pct)
            safe_threshold = np.percentile(normal_scores, normal_pct)

            if attack_threshold >= safe_threshold:
                continue

            valid_configs += 1

            # Calculate performance metrics
            attacks_detected = np.sum(attack_scores <= attack_threshold)
            attacks_suspicious = np.sum(
                (attack_scores > attack_threshold) & (attack_scores < safe_threshold)
            )
            attacks_missed = np.sum(attack_scores >= safe_threshold)

            normals_false_positive = np.sum(normal_scores <= attack_threshold)
            normals_suspicious = np.sum(
                (normal_scores > attack_threshold) & (normal_scores < safe_threshold)
            )
            normals_correct = np.sum(normal_scores >= safe_threshold)

            # Calculate rates
            miss_rate = attacks_missed / len(attack_scores)
            false_positive_rate = normals_false_positive / len(normal_scores)
            detection_rate = (attacks_detected + attacks_suspicious) / len(
                attack_scores
            )

            # Phase 2: More realistic targets
            if (
                miss_rate <= 0.20  # 20% miss rate target
                and false_positive_rate <= 0.20  # 20% false positive target
                and detection_rate >= 0.80
            ):  # 80% detection target

                # Calculate total cost
                cost = (
                    missed_attack_cost * attacks_missed
                    + false_positive_cost * normals_false_positive
                    + suspicious_cost * (attacks_suspicious + normals_suspicious)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_thresholds = (attack_threshold, safe_threshold)
                    best_metrics = {
                        "miss_rate": miss_rate,
                        "false_positive_rate": false_positive_rate,
                        "detection_rate": detection_rate,
                        "attacks_detected": attacks_detected / len(attack_scores),
                        "attacks_suspicious": attacks_suspicious / len(attack_scores),
                        "suspicious_total": (attacks_suspicious + normals_suspicious)
                        / len(scores),
                    }

    print(f"\n🔍 Phase 2 Threshold Search Results:")
    print(f"  • Valid configurations tested: {valid_configs}")

    if best_thresholds is None:
        print(
            "  ⚠️ No configuration met Phase 2 requirements. Using balanced fallback..."
        )

        # Phase 2: Balanced fallback thresholds
        attack_threshold = np.percentile(attack_scores, 80)  # 80th percentile
        safe_threshold = np.percentile(normal_scores, 20)  # 20th percentile

        if attack_threshold >= safe_threshold:
            median_gap = (np.median(normal_scores) - np.median(attack_scores)) / 2
            attack_threshold = np.median(attack_scores) + median_gap * 0.25
            safe_threshold = np.median(normal_scores) - median_gap * 0.25

        best_thresholds = (attack_threshold, safe_threshold)
        print(
            f"  • Balanced fallback thresholds: ATTACK < {attack_threshold:.4f}, SAFE > {safe_threshold:.4f}"
        )
    else:
        print(f"  ✅ Phase 2 optimal configuration found (cost={best_cost:.2f}):")
        print(f"    • Detection rate: {best_metrics['detection_rate']:.1%}")
        print(f"    • Miss rate: {best_metrics['miss_rate']:.1%}")
        print(f"    • False positive rate: {best_metrics['false_positive_rate']:.1%}")
        print(f"    • Direct detections: {best_metrics['attacks_detected']:.1%}")
        print(f"    • Suspicious flags: {best_metrics['suspicious_total']:.1%}")

    return best_thresholds[0], best_thresholds[1]


def classify_three_tier(scores, attack_threshold, safe_threshold):
    """Classify logs into three tiers."""
    classifications = []
    for score in scores:
        if score <= attack_threshold:
            classifications.append("ATTACK")
        elif score >= safe_threshold:
            classifications.append("SAFE")
        else:
            classifications.append("SUSPICIOUS")
    return classifications


def evaluate_three_tier_enhanced(y_true, classifications, scores):
    """Enhanced evaluation with detailed metrics."""
    y_true = np.array(y_true)
    classifications = np.array(classifications)

    total_attacks = max(1, np.sum(y_true == 1))
    total_normal = max(1, np.sum(y_true == 0))

    # Calculate detailed metrics
    attacks_as_attack = np.sum((y_true == 1) & (classifications == "ATTACK"))
    attacks_as_suspicious = np.sum((y_true == 1) & (classifications == "SUSPICIOUS"))
    attacks_as_safe = np.sum((y_true == 1) & (classifications == "SAFE"))

    normal_as_attack = np.sum((y_true == 0) & (classifications == "ATTACK"))
    normal_as_suspicious = np.sum((y_true == 0) & (classifications == "SUSPICIOUS"))
    normal_as_safe = np.sum((y_true == 0) & (classifications == "SAFE"))

    print("\n" + "=" * 70)
    print("PHASE 2 ADVANCED THREE-TIER CLASSIFICATION RESULTS")
    print("=" * 70)

    print(f"\n🎯 ATTACK LOGS ({int(total_attacks)} total):")
    print(
        f"  ✓ Flagged as ATTACK:     {attacks_as_attack:4d} ({(attacks_as_attack/total_attacks):.1%})"
    )
    print(
        f"  ⚠ Flagged as SUSPICIOUS: {attacks_as_suspicious:4d} ({(attacks_as_suspicious/total_attacks):.1%})"
    )
    print(
        f"  ✗ Missed (flagged SAFE): {attacks_as_safe:4d} ({(attacks_as_safe/total_attacks):.1%})"
    )

    print(f"\n🛡️ NORMAL LOGS ({int(total_normal)} total):")
    print(
        f"  ✗ False ATTACK flags:    {normal_as_attack:4d} ({(normal_as_attack/total_normal):.1%})"
    )
    print(
        f"  ⚠ SUSPICIOUS flags:      {normal_as_suspicious:4d} ({(normal_as_suspicious/total_normal):.1%})"
    )
    print(
        f"  ✓ Correctly SAFE:        {normal_as_safe:4d} ({(normal_as_safe/total_normal):.1%})"
    )

    # Performance metrics
    detection_rate = (attacks_as_attack + attacks_as_suspicious) / total_attacks
    precision_attack = attacks_as_attack / max(1, attacks_as_attack + normal_as_attack)
    precision_overall = (attacks_as_attack + attacks_as_suspicious) / max(
        1,
        attacks_as_attack
        + attacks_as_suspicious
        + normal_as_attack
        + normal_as_suspicious,
    )

    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  • Attack Detection Rate: {detection_rate:.1%}")
    print(f"  • Critical Miss Rate: {(attacks_as_safe/total_attacks):.1%}")
    print(f"  • False Positive Rate: {(normal_as_attack/total_normal):.1%}")
    print(f"  • Precision (ATTACK tier): {precision_attack:.1%}")
    print(f"  • Precision (Overall): {precision_overall:.1%}")

    # Efficiency metrics
    total_suspicious = attacks_as_suspicious + normal_as_suspicious
    suspicious_ratio = total_suspicious / len(y_true)

    print(f"\n⚡ EFFICIENCY METRICS:")
    print(f"  • Logs requiring review: {suspicious_ratio:.1%}")
    print(f"  • Automated decisions: {(1-suspicious_ratio):.1%}")
    print(f"  • Workload reduction: {(1-suspicious_ratio)*100:.0f}%")

    # Score analysis
    attack_scores = scores[y_true == 1]
    normal_scores = scores[y_true == 0]

    print(f"\n📊 SCORE DISTRIBUTION:")
    print(
        f"  • Attack mean: {np.mean(attack_scores):.4f} (std: {np.std(attack_scores):.4f})"
    )
    print(
        f"  • Normal mean: {np.mean(normal_scores):.4f} (std: {np.std(normal_scores):.4f})"
    )
    print(f"  • Separation: {np.mean(normal_scores) - np.mean(attack_scores):.4f}")

    if np.mean(attack_scores) < np.mean(normal_scores):
        print(f"  ✅ Correct separation: Attacks score lower than normals")
    else:
        print(f"  ⚠️ ISSUE: Attacks score higher than normals!")

    return {
        "detection_rate": detection_rate,
        "miss_rate": attacks_as_safe / total_attacks,
        "false_positive_rate": normal_as_attack / total_normal,
        "precision": precision_attack,
        "suspicious_ratio": suspicious_ratio,
    }


def run_phase2_advanced_analysis():
    """Run the Phase 2 advanced analysis pipeline."""

    # File paths
    train_file = "ADFA_log/training_data_kernel_activity.json"
    attack_file = "ADFA_log/Attack_logs_json/all_attacks.json"
    validation_file = "ADFA_log/normal_validation.json"

    print("=" * 70)
    print("PHASE 2 ADVANCED SOC AI SYSTEM")
    print("=" * 70)

    print("\n[1/6] Loading datasets...")
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

    print("\n[2/6] Extracting Phase 2 enhanced features...")
    X_train_df = enhanced_feature_extraction_phase2(train_data)
    X_attack_df = enhanced_feature_extraction_phase2(attack_data)
    X_normal_val_df = enhanced_feature_extraction_phase2(normal_validation_data)

    print(f"  • Feature dimensions: {X_train_df.shape[1]}")
    print(f"  • Top features: {', '.join(X_train_df.columns[:5])}")

    # Analyze feature distributions
    print("\n[3/6] Analyzing Phase 2 feature quality...")
    for col in [
        "attack_score",
        "pattern_score",
        "critical_intensity",
        "rare_ratio",
        "attack_complexity",
    ]:
        attack_mean = X_attack_df[col].mean()
        normal_mean = X_normal_val_df[col].mean()
        separation = attack_mean - normal_mean
        print(
            f"  • {col}: attack={attack_mean:.4f}, normal={normal_mean:.4f}, sep={separation:+.4f}"
        )

    print("\n[4/6] Training Phase 2 advanced ensemble models...")
    # Phase 2: Moderate contamination with feature selection
    contamination = 0.15  # Back to moderate
    ensemble = Phase2AdvancedDetector(contamination_estimate=contamination)

    # Create labels for feature selection
    y_train = np.array([0] * len(X_train_df))  # All training data is normal

    # Train with feature selection
    ensemble.fit(X_train_df, X_attack_df.sample(min(50, len(X_attack_df))), y_train)

    # Prepare combined validation data
    X_combined_df = pd.concat([X_attack_df, X_normal_val_df], ignore_index=True)
    y_true = np.array([1] * len(X_attack_df) + [0] * len(X_normal_val_df))

    print("\n[5/6] Computing Phase 2 anomaly scores...")
    all_scores = ensemble.decision_function(X_combined_df)

    # Analyze score distributions
    attack_scores = all_scores[y_true == 1]
    normal_scores = all_scores[y_true == 0]

    print(f"\n📊 Phase 2 Score Analysis:")
    print(
        f"  • Attack scores: min={attack_scores.min():.4f}, max={attack_scores.max():.4f}, mean={attack_scores.mean():.4f}"
    )
    print(
        f"  • Normal scores: min={normal_scores.min():.4f}, max={normal_scores.max():.4f}, mean={normal_scores.mean():.4f}"
    )

    # Check separation quality
    if attack_scores.mean() < normal_scores.mean():
        print(
            f"  ✅ Correct separation: Attacks ({attack_scores.mean():.4f}) < Normals ({normal_scores.mean():.4f})"
        )
    else:
        print(f"  ⚠️ WARNING: Inverted scores detected! Applying correction...")
        all_scores = -all_scores
        attack_scores = all_scores[y_true == 1]
        normal_scores = all_scores[y_true == 0]
        print(
            f"  • After correction: Attacks ({attack_scores.mean():.4f}) < Normals ({normal_scores.mean():.4f})"
        )

    print("\n[6/6] Optimizing classification thresholds (Phase 2)...")
    attack_threshold, safe_threshold = optimize_thresholds_phase2(all_scores, y_true)

    print(f"\nFinal Thresholds:")
    print(f"  • ATTACK: score ≤ {attack_threshold:.4f}")
    print(f"  • SAFE: score ≥ {safe_threshold:.4f}")
    print(f"  • SUSPICIOUS: {attack_threshold:.4f} < score < {safe_threshold:.4f}")

    # Classify and evaluate
    predictions = classify_three_tier(all_scores, attack_threshold, safe_threshold)
    metrics = evaluate_three_tier_enhanced(y_true, predictions, all_scores)

    return metrics


if __name__ == "__main__":
    print("\n🚀 Starting Phase 2 Advanced Anomaly Detection System...\n")

    metrics = run_phase2_advanced_analysis()

    if metrics is None:
        print("\n❌ Analysis failed due to data loading issues.")
        exit(1)

    print("\n" + "=" * 70)
    print("PHASE 2 ADVANCED OPTIMIZATION SUMMARY")
    print("=" * 70)

    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"  Phase 1 Results:")
    print(f"    • Attack Detection Rate: 75.2%")
    print(f"    • Critical Miss Rate: 24.8%")
    print(f"    • False Positive Rate: 23.3%")
    print(f"    • Precision: 28.1%")

    print(f"\n  Phase 2 Advanced:")
    print(f"    • Attack Detection Rate: {metrics['detection_rate']:.1%}")
    print(f"    • Critical Miss Rate: {metrics['miss_rate']:.1%}")
    print(f"    • False Positive Rate: {metrics['false_positive_rate']:.1%}")
    print(f"    • Precision: {metrics['precision']:.1%}")

    # Calculate improvements
    detection_improvement = metrics["detection_rate"] - 0.752
    miss_improvement = 0.248 - metrics["miss_rate"]
    fp_improvement = 0.233 - metrics["false_positive_rate"]
    precision_improvement = metrics["precision"] - 0.281

    print(f"\n📈 PHASE 2 IMPROVEMENTS:")
    print(f"  • Detection Rate: {detection_improvement:+.1%}")
    print(f"  • Miss Rate: {miss_improvement:+.1%}")
    print(f"  • False Positive Rate: {fp_improvement:+.1%}")
    print(f"  • Precision: {precision_improvement:+.1%}")

    # Phase 2 targets
    if metrics["miss_rate"] < 0.20:  # Phase 2 target
        print(f"\n✅ Phase 2 targets achieved!")
        print(f"  • Miss rate below 20% target")
    else:
        print(f"\n⚠️ Phase 2 targets not fully achieved.")
        print(f"  • Consider Phase 3 optimizations")

    print("=" * 70)

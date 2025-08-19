import json
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def raw_feature_extraction(logs):
    """Raw feature extraction without any weights or attack-specific logic."""

    # Collect all unique syscalls across all logs
    all_syscalls = set()
    for log in logs:
        syscalls = log.get("kernel", {}).get("syscall_counts", {})
        all_syscalls.update(syscalls.keys())

    # Sort syscalls for consistent feature order
    all_syscalls = sorted(list(all_syscalls))
    print(f"  • Total unique syscalls found: {len(all_syscalls)}")

    features = []

    for log in logs:
        syscalls = log.get("kernel", {}).get("syscall_counts", {})
        total = max(1, sum(syscalls.values()))

        # Create feature vector with raw syscall counts (normalized by total)
        feature_vector = []
        for syscall in all_syscalls:
            count = syscalls.get(syscall, 0)
            normalized_count = count / total
            feature_vector.append(normalized_count)

        # Add basic statistics without any attack-specific logic
        unique_syscalls = len(syscalls)
        total_activity = np.log1p(total)

        # Add these as additional features
        feature_vector.extend([unique_syscalls, total_activity])

        features.append(feature_vector)

    # Create column names
    columns = all_syscalls + ["unique_syscalls", "total_activity"]

    return pd.DataFrame(features, columns=columns)


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for anomaly detection."""

    def __init__(self, input_dim, encoding_dim=20):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, encoding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, input_dim),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(model, train_loader, num_epochs=100, learning_rate=0.001):
    """Train the autoencoder on normal data."""

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train_losses = []

    print(f"  • Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)

            # Forward pass
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return train_losses


def compute_reconstruction_error(model, data_loader):
    """Compute reconstruction error for anomaly detection."""

    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch_x in data_loader:
            batch_x = batch_x[0].to(device)
            reconstructed = model(batch_x)

            # Compute MSE for each sample
            mse = torch.mean((batch_x - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(mse.cpu().numpy())

    return np.array(reconstruction_errors)


def evaluate_autoencoder_results(y_true, reconstruction_errors):
    """Evaluate autoencoder results with detailed classification breakdown."""

    y_true = np.array(y_true)
    reconstruction_errors = np.array(reconstruction_errors)

    # Calculate percentiles for threshold selection
    attack_errors = reconstruction_errors[y_true == 1]
    normal_errors = reconstruction_errors[y_true == 0]

    print(f"\n📊 Autoencoder Error Analysis:")
    print(
        f"  • Attack errors: min={attack_errors.min():.6f}, max={attack_errors.max():.6f}, mean={attack_errors.mean():.6f}"
    )
    print(
        f"  • Normal errors: min={normal_errors.min():.6f}, max={normal_errors.max():.6f}, mean={normal_errors.mean():.6f}"
    )
    print(
        f"  • Error separation: {np.mean(attack_errors) - np.mean(normal_errors):.6f}"
    )

    # Simple threshold-based evaluation
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n🔍 Autoencoder Threshold Analysis:")
    print(
        f"{'Threshold':<10} {'Detection Rate':<15} {'False Positive Rate':<20} {'Precision':<10}"
    )
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    best_predictions = None

    for threshold in thresholds:
        # Use percentile-based threshold
        error_threshold = np.percentile(reconstruction_errors, threshold * 100)

        # Classify (higher error = more anomalous)
        predictions = (reconstruction_errors >= error_threshold).astype(int)

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
                "threshold": error_threshold,
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

    return best_metrics, reconstruction_errors


def run_autoencoder_analysis():
    """Run autoencoder analysis on the dataset."""

    # File paths
    train_file = "ADFA_log/training_data_kernel_activity.json"
    attack_file = "ADFA_log/Attack_logs_json/all_attacks.json"
    validation_file = "ADFA_log/normal_validation.json"

    print("=" * 60)
    print("AUTOENCODER ANOMALY DETECTION")
    print("=" * 60)

    print("\n[1/5] Loading datasets...")
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

    print("\n[2/5] Extracting raw features...")
    # Combine all data to get complete syscall vocabulary
    all_data = train_data + attack_data + normal_validation_data
    X_all_df = raw_feature_extraction(all_data)

    # Split back into datasets
    X_train_df = X_all_df.iloc[: len(train_data)]
    X_attack_df = X_all_df.iloc[len(train_data) : len(train_data) + len(attack_data)]
    X_normal_val_df = X_all_df.iloc[len(train_data) + len(attack_data) :]

    print(f"  • Feature dimensions: {X_train_df.shape[1]}")
    print(f"  • Features: Raw syscall counts + basic stats")

    print("\n[3/5] Preprocessing data...")
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_attack_scaled = scaler.transform(X_attack_df)
    X_normal_val_scaled = scaler.transform(X_normal_val_df)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_attack_tensor = torch.FloatTensor(X_attack_scaled)
    X_normal_val_tensor = torch.FloatTensor(X_normal_val_scaled)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor)
    # Use smaller batch size to avoid BatchNorm issues
    batch_size = min(16, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n[4/5] Training autoencoder...")
    # Initialize and train autoencoder
    input_dim = X_train_df.shape[1]
    encoding_dim = min(20, input_dim // 4)  # Adaptive encoding dimension

    model = SimpleAutoencoder(input_dim, encoding_dim).to(device)
    print(f"  • Input dimensions: {input_dim}")
    print(f"  • Encoding dimensions: {encoding_dim}")

    train_losses = train_autoencoder(model, train_loader, num_epochs=100)

    print("\n[5/5] Evaluating on validation data...")
    # Prepare combined validation data
    X_combined_tensor = torch.cat([X_attack_tensor, X_normal_val_tensor], dim=0)
    combined_dataset = TensorDataset(X_combined_tensor)
    # Use smaller batch size for evaluation too
    eval_batch_size = min(16, len(combined_dataset))
    combined_loader = DataLoader(
        combined_dataset, batch_size=eval_batch_size, shuffle=False
    )

    # Compute reconstruction errors
    reconstruction_errors = compute_reconstruction_error(model, combined_loader)

    # Create labels
    y_true = np.array([1] * len(X_attack_df) + [0] * len(X_normal_val_df))

    # Evaluate results
    best_metrics, errors = evaluate_autoencoder_results(y_true, reconstruction_errors)

    if best_metrics:
        print(f"\n" + "=" * 60)
        print("AUTOENCODER RESULTS SUMMARY")
        print("=" * 60)

        print(f"\n🎯 Best Performance (Threshold: {best_metrics['threshold']:.6f}):")
        print(f"  • Detection Rate: {best_metrics['detection_rate']:.1%}")
        print(f"  • False Positive Rate: {best_metrics['false_positive_rate']:.1%}")
        print(f"  • Precision: {best_metrics['precision']:.1%}")
        print(f"  • F1 Score: {best_metrics['f1_score']:.3f}")

        # Calculate additional metrics
        attack_errors = errors[y_true == 1]
        normal_errors = errors[y_true == 0]

        print(f"\n📊 Error Distribution:")
        print(
            f"  • Attack mean: {np.mean(attack_errors):.6f} (std: {np.std(attack_errors):.6f})"
        )
        print(
            f"  • Normal mean: {np.mean(normal_errors):.6f} (std: {np.std(normal_errors):.6f})"
        )
        print(f"  • Separation: {np.mean(attack_errors) - np.mean(normal_errors):.6f}")

        if np.mean(attack_errors) > np.mean(normal_errors):
            print(f"  ✅ Good separation: Attacks have higher reconstruction errors")
        else:
            print(f"  ⚠️ Poor separation: Attacks have lower reconstruction errors")

    return best_metrics


if __name__ == "__main__":
    print("\n🚀 Starting Autoencoder Analysis...\n")

    metrics = run_autoencoder_analysis()

    if metrics is None:
        print("\n❌ Analysis failed due to data loading issues.")
        exit(1)

    print("\n" + "=" * 60)
    print("AUTOENCODER ANALYSIS COMPLETE")
    print("=" * 60)

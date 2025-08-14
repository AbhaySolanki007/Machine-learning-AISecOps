import os
import json
from datetime import datetime
from collections import defaultdict

# ==== CONFIG ====
MAPPING_FILE = r"C:\Users\dell\Downloads\ADFA_log\ADFA-LD\ADFA-LD+Syscall+List.txt"
ATTACK_DATA_DIR = r"C:\Users\dell\Downloads\ADFA_log\ADFA-LD\Attack_Data_Master"
OUTPUT_DIR = r"C:\Users\dell\Downloads\ADFA_log\OCSF_Attacks"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== STEP 1: Load syscall mapping ====
syscall_map = {}
with open(MAPPING_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        # Expecting lines like: "#define __NR_read 63"
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("__NR_"):
            try:
                num = str(int(parts[2]))  # syscall number as string
                name = parts[1].replace("__NR_", "")  # remove prefix
                syscall_map[num] = name
            except ValueError:
                continue

# ==== STEP 2: Prepare storage ====
attack_groups = defaultdict(list)  # {attack_type: [OCSF_events]}
all_attacks = []  # combined list

# ==== STEP 3: Process attack data ====
for folder_name in os.listdir(ATTACK_DATA_DIR):
    folder_path = os.path.join(ATTACK_DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Extract base attack name before the underscore (e.g., "Web_Shell" from "Web_Shell_1")
    attack_type = "_".join(folder_name.split("_")[:-1])
    if not attack_type:
        attack_type = folder_name  # fallback if no underscore

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                syscalls = f.read().strip().split()

            # Map syscall numbers to names and count occurrences
            syscall_counts = {}
            for num in syscalls:
                name = syscall_map.get(num, f"syscall_{num}")
                syscall_counts[name] = syscall_counts.get(name, 0) + 1

            # Build OCSF Kernel Activity Event
            event = {
                "class_uid": 1003,
                "class_name": "Kernel Activity",
                "category_uid": 1,
                "category_name": "System Activity",
                "time": datetime.utcnow().isoformat() + "Z",
                "activity_id": 1,
                "activity_name": "System Call Trace",
                "type_uid": 1,
                "type_name": "System Call",
                "os": {"type": "Linux"},
                "metadata": {"filename": filename, "trace_length": len(syscalls)},
                "kernel": {"syscall_counts": syscall_counts},
                "severity_id": 1,
                "severity": "Informational",
                "status_id": 1,
                "status": "Success",
                "label": "attack",
                "attack_type": attack_type,
            }

            attack_groups[attack_type].append(event)
            all_attacks.append(event)

# ==== STEP 4: Save JSON files per attack type ====
for attack_type, events in attack_groups.items():
    output_file = os.path.join(OUTPUT_DIR, f"{attack_type.lower()}.json")
    with open(output_file, "w") as f:
        json.dump(events, f, indent=4)
    print(f"✅ Saved {attack_type} attacks to {output_file}")

# ==== STEP 5: Save combined attacks JSON ====
all_attacks_file = os.path.join(OUTPUT_DIR, "all_attacks.json")
with open(all_attacks_file, "w") as f:
    json.dump(all_attacks, f, indent=4)

print(f"✅ Saved all attacks to {all_attacks_file}")

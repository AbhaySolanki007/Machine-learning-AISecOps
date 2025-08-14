import os
import json
from datetime import datetime

# ==== CONFIG ====
MAPPING_FILE = r"C:\Users\dell\Downloads\ADFA_log\ADFA-LD\ADFA-LD+Syscall+List.txt"
TRAINING_DATA_DIR = r"C:\Users\dell\Downloads\ADFA_log\ADFA-LD\Training_Data_Master"
OUTPUT_FILE = r"C:\Users\dell\Downloads\ADFA_log\training_data_kernel_activity.json"

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

# ==== STEP 2: Process training data ====
ocsf_events = []
for filename in os.listdir(TRAINING_DATA_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(TRAINING_DATA_DIR, filename)
        with open(filepath, "r") as f:
            syscalls = f.read().strip().split()

        # Map syscall numbers to names and count occurrences
        syscall_counts = {}
        for num in syscalls:
            name = syscall_map.get(num, f"syscall_{num}")  # fallback if not found
            syscall_counts[name] = syscall_counts.get(name, 0) + 1

        # ==== STEP 3: Build OCSF Kernel Activity Event ====
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
            "label": "normal",
        }

        ocsf_events.append(event)

# ==== STEP 4: Save to JSON file ====
with open(OUTPUT_FILE, "w") as f:
    json.dump(ocsf_events, f, indent=4)

print(f"✅ OCSF Kernel Activity JSON saved to {OUTPUT_FILE}")

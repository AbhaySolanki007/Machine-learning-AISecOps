import os
import json
import re
from collections import Counter

# --- STEP 1: SET YOUR FOLDER PATHS HERE ---
# IMPORTANT: Replace the paths below with the correct paths on your system.
# Use the 'r' before the string to handle backslashes correctly.

# Path to the main folder where you want the output to be saved.
base_output_path = r"C:\Users\dell\Downloads\ADFA_log"

# Path to the directory containing 'ADFA-LD+Syscall+List.txt' and 'Training_Data_Master'
# This is often a subdirectory within your main log folder.
adfa_ld_path = r"C:\Users\dell\Downloads\ADFA_log\ADFA-LD"


# --- STEP 2: DEFINE FILE AND DIRECTORY NAMES ---
syscall_list_filename = "ADFA-LD+Syscall+List.txt"
training_data_dirname = "Training_Data_Master"
output_json_filename = "training_data_named_features.json"


# --- Main Logic (No changes needed below this line) ---


def parse_syscall_list(syscall_file_path):
    """Parses the syscall list file and returns a number-to-name mapping."""
    syscall_map = {}
    # Regex to find lines like: #define __NR_lsetxattr 6
    # It captures the name ('lsetxattr') and the number ('6')
    define_pattern = re.compile(r"#define\s+__NR_([a-zA-Z0-9_]+)\s+([0-9]+)")

    try:
        with open(syscall_file_path, "r") as f:
            for line in f:
                match = define_pattern.match(line)
                if match:
                    name = match.group(1)
                    number = int(match.group(2))
                    syscall_map[number] = name
    except FileNotFoundError:
        print(
            f"--- FATAL ERROR ---: Syscall list file not found at '{syscall_file_path}'"
        )
        return None

    print(f"Successfully parsed {len(syscall_map)} system calls from the list.")
    return syscall_map


def process_and_translate_data():
    """Finds, processes, translates, and saves training data to a single JSON file."""
    syscall_file_path = os.path.join(adfa_ld_path, syscall_list_filename)
    training_dir = os.path.join(adfa_ld_path, training_data_dirname)
    output_file = os.path.join(base_output_path, output_json_filename)

    # First, create the decoder ring
    syscall_map = parse_syscall_list(syscall_file_path)
    if syscall_map is None:
        return  # Stop if we can't find the syscall list

    if not os.path.isdir(training_dir):
        print(
            f"--- FATAL ERROR ---: Training data directory not found at '{training_dir}'"
        )
        return

    all_traces = []
    file_list = [f for f in os.listdir(training_dir) if f.endswith(".txt")]
    print(f"Found {len(file_list)} .txt files to process. Starting...")

    for filename in file_list:
        file_path = os.path.join(training_dir, filename)
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                if not content:
                    continue

                syscall_sequence = [int(num) for num in content.split()]

                # --- TRANSLATION STEP ---
                # Translate the numbers to names using the map
                # If a number is not in our map, use "syscall_NUMBER" as a fallback
                named_counts = Counter(
                    syscall_map.get(num, f"syscall_{num}") for num in syscall_sequence
                )

                trace_data = {
                    "filename": filename,
                    "label": "normal",
                    "trace_length": len(syscall_sequence),
                    "syscall_counts": dict(named_counts),
                }
                all_traces.append(trace_data)
        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")

    # Save the consolidated list to the final JSON file
    try:
        with open(output_file, "w") as f:
            json.dump(all_traces, f, indent=4)

        print("\n--- SUCCESS! ---")
        print(f"Successfully processed {len(all_traces)} files with name translation.")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"\n--- ERROR ---: Could not write to the output file. Error: {e}")


if __name__ == "__main__":
    process_and_translate_data()

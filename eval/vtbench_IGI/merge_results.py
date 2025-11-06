
import json
import os
import glob
import argparse 

def merge_json_files(results_dir, merged_output_file):
    all_results = []

    file_pattern = os.path.join(results_dir, "results_worker_*.json")
    worker_files = glob.glob(file_pattern)

    if not worker_files:
        print(f"Error: No result files found in '{results_dir}'.")
        return

    print(f"Found {len(worker_files)} result files to merge.")
    worker_files.sort()

    for file_path in worker_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.extend(data)
                print(f"  - Merged {len(data)} records from {os.path.basename(file_path)}")
        except json.JSONDecodeError:
            print(f"  - Warning: JSON Decode Error in {os.path.basename(file_path)}. 跳过。")
        except Exception as e:
            print(f"  - Error reading file {os.path.basename(file_path)}: {e}")

    print(f"\nTotal records merged: {len(all_results)}")
    print(f"Saving merged results to '{merged_output_file}'...")
    
    os.makedirs(os.path.dirname(merged_output_file), exist_ok=True)
    with open(merged_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("Merging complete! ✨")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge JSON results from parallel workers.")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    
    args = parser.parse_args()
    merge_json_files(args.results_dir, args.output_file)

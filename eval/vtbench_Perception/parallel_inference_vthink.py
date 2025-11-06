# parallel_inference.py

import torch
import os
import json
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from tqdm import tqdm
from utils_vthink import run_evaluation # Assuming you have this utility function

def main():
    # --- 1. Setup Argument Parser ---
    # This will read arguments passed from the shell script
    parser = argparse.ArgumentParser(description="Run parallel inference for a VL model.")
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID for logging. The actual device is set by CUDA_VISIBLE_DEVICES in the shell script.')
    parser.add_argument('--worker_id', type=int, required=True, help='The ID of this worker (e.g., 0, 1, 2, ...)')
    parser.add_argument('--num_workers', type=int, required=True, help='The total number of workers.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the partial results.')
    # You can also make these configurable
    parser.add_argument('--model_path', type=str, required=True,)
    parser.add_argument('--data_json_path', type=str,required=True,)
    parser.add_argument('--aux_outpath', type=str, required=True, help='Directory to save the output files.')

    args = parser.parse_args()

    # --- Configuration from Args ---
    WORKER_ID = args.worker_id
    NUM_WORKERS = args.num_workers
    OUTPUT_DIR = args.output_dir
    MODEL_PATH = args.model_path
    DATA_JSON_PATH = args.data_json_path
    RESULTS_SAVE_PATH = os.path.join(OUTPUT_DIR, f"results_worker_{WORKER_ID}.json")

    print(f"--- Worker {WORKER_ID}/{NUM_WORKERS} Started ---")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Data Path: {DATA_JSON_PATH}")
    print(f"Saving results to: {RESULTS_SAVE_PATH}")


    # --- 2. Load Model and Processor ---
    # The CUDA_VISIBLE_DEVICES environment variable set by the shell script
    # will ensure this process only sees and uses the assigned GPU.
    print(f"[Worker {WORKER_ID}] Loading model and processor...")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto", # "auto" works perfectly with CUDA_VISIBLE_DEVICES
        config=config
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"[Worker {WORKER_ID}] Model and processor loaded successfully. ✅")


    # --- 3. Load Input Data from JSON ---
    print(f"[Worker {WORKER_ID}] Loading data from {DATA_JSON_PATH}...")
    with open(DATA_JSON_PATH, 'r', encoding='utf-8') as f:
        full_data_list = json.load(f)
    print(f"[Worker {WORKER_ID}] Full dataset has {len(full_data_list)} total items.")

    # --- ✅ CORE CHANGE: Distribute Data Among Workers ---
    # Each worker takes a slice of the data.
    # Worker 0 gets items 0, 8, 16, ...
    # Worker 1 gets items 1, 9, 17, ...
    data_to_process = full_data_list[WORKER_ID::NUM_WORKERS]
    print(f"[Worker {WORKER_ID}] Assigned {len(data_to_process)} items to process.")


    # --- 4. Run Batch Evaluation on Assigned Data ---
    all_results = []
    print(f"[Worker {WORKER_ID}] Starting batch evaluation...")

    # Using tqdm to show a progress bar for this specific worker
    for item in tqdm(data_to_process, desc=f"Worker {WORKER_ID} Evaluating", position=WORKER_ID):
        import ast

# Your original line - we start with this
        question_data = item.get("perception_question")

        question_text = "" 

        if isinstance(question_data, dict):
            question_keys = [key for key in question_data if key.startswith('question')]
            try:
                question_keys.sort(key=lambda k: int(k[len('question'):]))
                
                extracted_questions = [question_data[key] for key in question_keys]
                
                question_text = "; ".join(extracted_questions)

            except ValueError:
                print("error")
            
        else:
            # if isinstance(question_data, str):
            #     try:
            #         question_data = ast.literal_eval(question_data)
            #         # ... 此处重复上面的 if isinstance(question_data, dict) 逻辑
            #     except (ValueError, SyntaxError):
            #         print(f"无法解析字符串: {question_data}")
            pass

        image_path = item.get("image_path")
        
        # Construct the full, normalized image path
        if not question_text or not image_path:
            print(f"[Worker {WORKER_ID}] Skipping item due to missing 'question3' or 'iimage_path': {item}")
            continue

        try:
            final_assistant_response, final_answer, aux_path = run_evaluation(question_text, image_path, args.aux_outpath, model, processor)
            
            result_entry = {
                "image_path": image_path,
                "question": question_text,
                "aux_path": aux_path ,
                "model_response": final_assistant_response,
                "final_answer": final_answer,
                "original_item": item 
            }
            all_results.append(result_entry)

        except Exception as e:
            print(f"[Worker {WORKER_ID}] An error occurred while processing {image_path}: {e}")
            error_entry = {
                "image_path": image_path,
                "question": question_text,
                "aux_path": aux_path ,
                "model_response": final_assistant_response,
                "final_answer": final_answer,
                "original_item": item 
            }
            all_results.append(error_entry)

    # --- 5. Save Partial Results for This Worker ---
    print(f"\n[Worker {WORKER_ID}] Evaluation finished. Saving {len(all_results)} results to {RESULTS_SAVE_PATH}...")
    with open(RESULTS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"--- Worker {WORKER_ID} Done! ✨ ---")


if __name__ == "__main__":
    main()

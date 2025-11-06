import torch
import os
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from tqdm import tqdm
from utils import run_evaluation
import base64
import pandas as pd
import string
import argparse 

DECODED_IMAGES_DIR = "./decoded_images_wemath/" 

def setup_directories():
    if not os.path.exists(DECODED_IMAGES_DIR):
        os.makedirs(DECODED_IMAGES_DIR)
        print(f"Created directory: {DECODED_IMAGES_DIR}")

def read_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def decode_base64_to_image(b64_string: str, output_path: str):
    try:
        image_data = base64.b64decode(b64_string)
        with open(output_path, 'wb') as f:
            f.write(image_data)
    except Exception as e:
        print(f"Error decoding or saving image {output_path}: {e}")

def build_wemath_prompt(data_row: pd.Series, index: int) -> tuple[str, str]:
    image_b64_string = data_row['image']
    image_filename = f"{index}.png" 
    full_image_path = os.path.join(DECODED_IMAGES_DIR, image_filename)

    if not read_ok(full_image_path):
        decode_base64_to_image(image_b64_string, full_image_path)
    
    question = data_row['question']
    options = {
        cand: data_row[cand]
        for cand in string.ascii_uppercase
        if cand in data_row and not pd.isna(data_row[cand])
    }
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    hint = data_row['hint'] if 'hint' in data_row and not pd.isna(data_row['hint']) else None
    prompt_text = ""
    if hint is not None:
        prompt_text += f'Hint: {hint}\n\n'
    prompt_text += f'Question: {question}\n\n'
    if len(options):
        prompt_text += options_prompt
    prompt_text += "\nPlease provide the final answer in the format <answer>X</answer>, where X is the correct option letter."

    return prompt_text.strip(), full_image_path

def main():
    parser = argparse.ArgumentParser(description="Run parallel inference on the WeMath dataset.")
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID for logging. The actual device is set by CUDA_VISIBLE_DEVICES in the shell script.')
    parser.add_argument('--num_workers', type=int, required=True, help='Total number of parallel workers.')
    parser.add_argument('--worker_id', type=int, required=True, help='ID of this specific worker (from 0 to num_workers-1).')
    parser.add_argument('--output_dir', type=str, default='./parallel_results_base', help='Directory to save the output files.')
    parser.add_argument('--model_path', type=str, default='./parallel_results_base', help='Directory to save the output files.')
    parser.add_argument('--aux_outpath', type=str, default='./base', help='Directory to save the output files.')
    parser.add_argument('--data_json_path', type=str, required=True)
    args = parser.parse_args()

    print(f"[Worker {args.worker_id}] Assigned to GPU: {args.gpu_id}. `device_map='auto'` will use the GPU made visible by the launcher.")


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    setup_directories()

    # ======
    print(f"[Worker {args.worker_id}] Loading model and processor...")
    config = AutoConfig.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        config=config
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print(f"[Worker {args.worker_id}] Model and processor loaded successfully. ✅")

    # --- 2. Load and Split Data ---
    print(f"[Worker {args.worker_id}] Loading and splitting data...")
    wemath_data = pd.read_csv(args.data_json_path, sep='\t')
    
    total_samples = len(wemath_data)
    chunk_size = (total_samples + args.num_workers - 1) // args.num_workers 
    start_index = args.worker_id * chunk_size
    end_index = min(start_index + chunk_size, total_samples)
    
    if start_index >= total_samples:
        print(f"[Worker {args.worker_id}] No data to process. Exiting.")
        return

    data_chunk = wemath_data.iloc[start_index:end_index]
    print(f"[Worker {args.worker_id}] Processing data slice: {start_index} to {end_index-1} ({len(data_chunk)} samples)")

    # --- 3. Run Batch Evaluation on the Chunk ---
    all_results = []
    print(f"[Worker {args.worker_id}] Starting batch evaluation...")

    for index, item in tqdm(data_chunk.iterrows(), total=len(data_chunk), desc=f"Worker {args.worker_id} Progress"):
        question_text, image_path = build_wemath_prompt(item, index)

        if not question_text or not image_path:
            print(f"Skipping item due to missing data: index {index}")
            continue
            
        try:
            final_assistant_response, final_answer,a = run_evaluation(question_text, image_path, args.aux_outpath, model, processor)
            result_entry = {
                "original_index": index,
                "image_path": image_path,
                "question": question_text,
                "model_response": final_assistant_response,
                "final_answer": final_answer,
                "original_item": item.to_dict()
            }
            all_results.append(result_entry)
        except Exception as e:
            print(f"[Worker {args.worker_id}] An error occurred while processing index {index}: {e}")
            error_entry = {
                "original_index": index,
                "image_path": image_path,
                "question": question_text,
                "error": str(e)
            }
            all_results.append(error_entry)

    output_path = os.path.join(args.output_dir, f"results_worker_{args.worker_id}.json")
    print(f"\n[Worker {args.worker_id}] Evaluation finished. Saving {len(all_results)} results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"[Worker {args.worker_id}] All done! ✨")

if __name__ == '__main__':
    main()

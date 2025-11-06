import torch
import os
import json
from transformers import  AutoProcessor, AutoConfig, AutoModelForCausalLM,Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
from utils import run_evaluation

import pandas as pd 
import string
import argparse
def load_jsonl_data(data_path: str) -> list[dict]:
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def build_prompt(data_row: dict) -> tuple[str, str]:
    image_path = data_row['image_path']

    # 1. Split the path into directory and full filename
    dir_part, file_part = os.path.split(image_path)
    # dir_part = 'some/dir', file_part = '0.png'

    # 2. Split the filename into base and extension
    file_base, file_ext = os.path.splitext(file_part)
    # file_base = '0', file_ext = '.png'

    # 3. Strip leading zeros from the base name
    new_file_base = file_base.lstrip('0')
    # new_file_base = '' (This is the problem you want to fix)

    # 4. === FIX: Check if stripping left an empty string ===
    #    If the base was '0' or '00', new_file_base is now ''.
    #    This line resets it back to '0'.
    if not new_file_base:
        new_file_base = '0'
    # new_file_base = '0'

    # 5. Set your new extension
    new_ext = '.jpg'

    # 6. Rebuild the full name and path
    new_full_file_name = new_file_base + new_ext
    # new_full_file_name = '0.jpg'

    processed_image_path = os.path.join(dir_part, new_full_file_name)
    # processed_image_path = 'some/dir/0.jpg'

    base_dir = "./"
    image_path = os.path.join(base_dir, processed_image_path)
    # image_path = './some/dir/0.jpg'
    question = data_row['question']

    if not os.path.exists(image_path):
        print(f"no image")

    prompt = ''
    prompt_text = ''
    prompt += question
    prompt += "\nSolve the complex visual logical reasoning problem through step-by-step reasoning."
    prompt += "Think about the reasoning process first "
    prompt_text += f'Question: {prompt}\n\n'
    prompt_text += "\nPlease provide the final answer in the format <answer>X</answer>"

    return prompt_text.strip(), image_path

def main():
    parser = argparse.ArgumentParser(description="Run parallel inference for a VL model.")
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID for logging. The actual device is set by CUDA_VISIBLE_DEVICES in the shell script.')
    parser.add_argument('--worker_id', type=int, required=True, help='The ID of this worker (e.g., 0, 1, 2, ...)')
    parser.add_argument('--num_workers', type=int, required=True, help='The total number of workers.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the partial results.')
    # You can also make these configurable
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_json_path', type=str, required=True)
    parser.add_argument('--aux_outpath', type=str, required=True, help='Directory to save the output files.')

    args = parser.parse_args()
    print(f"[Worker {args.worker_id}] Assigned to GPU: {args.gpu_id}. `device_map='auto'` 将使用启动器可见的 GPU。")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    print(f"[Worker {args.worker_id}] Loading model and processor from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        config=config
    )
    processor = AutoProcessor.from_pretrained(args.model_path,trust_remote_code=True)
    print(f"[Worker {args.worker_id}] Model and processor loaded successfully. ✅")
    print(f"[Worker {args.worker_id}] Model and processor loaded successfully. ✅")

    print(f"[Worker {args.worker_id}] Loading and splitting data from {args.data_json_path}...")
    all_data = load_jsonl_data(args.data_json_path)

    total_samples = len(all_data)
    chunk_size = (total_samples + args.num_workers - 1) // args.num_workers 
    start_index = args.worker_id * chunk_size
    end_index = min(start_index + chunk_size, total_samples)
    
    if start_index >= total_samples:
        print(f"[Worker {args.worker_id}] No data to process. Exiting.")
        return


    data_chunk = all_data[start_index:end_index]
    print(f"[Worker {args.worker_id}] Processing data slice: index {start_index} to {end_index-1} ({len(data_chunk)} samples)")

    all_results = []
    print(f"[Worker {args.worker_id}] Starting batch evaluation...")

    for i, item in enumerate(tqdm(data_chunk, total=len(data_chunk), desc=f"Worker {args.worker_id} Progress")):
        original_index = start_index + i
        question_text, image_path = build_prompt(item)

        if not question_text or not image_path:
            print(f"Skipping item due to missing data: original index {original_index}")
            continue
            
        try:
            final_assistant_response, final_answer = run_evaluation(question_text, image_path, args.aux_outpath, model, processor)
            result_entry = {
                "original_index": original_index,
                "image_path": image_path,
                "question": question_text,
                "model_response": final_assistant_response,
                "final_answer": final_answer,
                "original_item": item 
            }
            all_results.append(result_entry)
        except Exception as e:
            print(f"[Worker {args.worker_id}] An error occurred while processing original index {original_index}: {e}")
            error_entry = {
                "original_index": original_index,
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

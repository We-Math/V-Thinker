import json
import os
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.
Be strict!!!

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1 B. Can not determine.\n
[Standard Answer]: A
[Model_answer] : \\frac{{x^2}}{{81}} + \\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """

# Qwen2-VL
def run_qwen2_vl(questions, ground_truths, extractions,model_path):
    # ... (function content is fine) ...
    model_name = "Qwen3-30B-A3B-Instruct-2507"
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        max_num_seqs=1,
        tensor_parallel_size=8,
    )
    prompts = []
    for question, ground_truth, extraction in zip(questions, ground_truths, extractions):
        full_prompt = demo_prompt_score.format(
            question=question, 
            gt=ground_truth, 
            extraction=extraction
        )

        prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                  f"<|im_start|>user\n{full_prompt}<|im_end|>\n"
                  "<|im_start|>assistant\n")
        # print(prompt) # Printing too much can slow things down
        prompts.append(prompt)
    stop_token_ids = None
    return llm, prompts, stop_token_ids

def get_multi_modal_input(args, data):
    questions = []
    gts = []
    extractions = []
    valid_indices = []

    for index, item in enumerate(data):
        if "original_item" in item and "interactive_reasoning_answer" in item["original_item"] and item["original_item"]["interactive_reasoning_answer"] is not None:
            questions.append(item["original_item"]["interactive_reasoning_question"])
            gts.append(item["original_item"]["interactive_reasoning_answer"])
            extractions.append(item["Extracted answer"])
            valid_indices.append(index)
    return questions, gts, extractions, valid_indices

model_example_map = {
    "qwen2_vl": run_qwen2_vl,
    "qwen2_5": run_qwen2_vl
}

def main(args):
    if True:
        model = args.model_type
        if model not in model_example_map:
            raise ValueError(f"Model type {model} is not supported.")
        
        json_path = args.json_path

        # <-- BEST PRACTICE: Specify encoding when reading
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions, ground_truths, extractions, valid_indices = get_multi_modal_input(args, data)
        llm, prompts, stop_token_ids = model_example_map[model](questions, ground_truths, extractions,args.model_path)

        sampling_params = SamplingParams(temperature=0, max_tokens=1024, stop_token_ids=stop_token_ids)
        
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        for i, output in enumerate(tqdm(outputs, desc="Processing results")):
            original_data_index = valid_indices[i]
            generated_text = output.outputs[0].text
            data[original_data_index]['judge'] = generated_text

        result_file = args.result_file
        
        # <-- FIX: Specify encoding and ensure_ascii=False when writing
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    else:
        result_file = args.result_file
    def calculate_accuracy(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter items that have a 'judge' key to avoid errors
        judged_data = [inst for inst in data if 'judge' in inst]
        if not judged_data:
            print("No items with 'judge' key found.")
            return

        total = len(judged_data)
        exact_match_count = 0
        
        for inst in judged_data:
            
            judgement = None
            if 'judge' in inst:
                judge = inst.get("judge", "") # Use .get for safety
                judge_strip = judge.strip()
                if judge_strip.startswith("0") or judge_strip.startswith("1"):
                    judgement = judge_strip[0]
                elif "Judgement:" in judge:
                     # More robust extraction
                    match = re.search(r'Judgement:\s*([01])', judge, re.IGNORECASE)
                    if match:
                        judgement = match.group(1)
                
                if judgement == "1":
                    exact_match_count += 1
                    
        # Calculate accuracy based on items that were actually judged
        exact_accuracy = (exact_match_count / total * 100) if total > 0 else 0
        print(f"{exact_match_count} / {total}")
        print(f"acc: {exact_accuracy:.2f}%") # Using f-string for better formatting

    calculate_accuracy(result_file)


if __name__ == "__main__":
    # ... (your argument parsing code remains the same) ...
    model = "qwen2_5"
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models for text generation'
    )
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default=model,
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help='Number of prompts to run.')
    parser.add_argument('--modality',
                        type=str,
                        default="image",
                        choices=["image", "video"],
                        help='Modality of the input data.')
    parser.add_argument('--json-path',
                        type=str,
                        required=True,
                        help='Path to the JSON file containing questions and images.')
    parser.add_argument('--result-file',
                        type=str,
                        required=True,
                        help='Path to save the extraction results.')
    parser.add_argument("--model-path", type=str,
                        required=True)

    args = parser.parse_args()
    main(args)

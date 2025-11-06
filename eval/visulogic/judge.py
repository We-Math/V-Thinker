from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import json
from PIL import Image
import os
import pdb
from collections import defaultdict
from tqdm import tqdm
import random
import copy
import re
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
demo_prompt_score = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.

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
    for item in data:
        if "original_item" in item and "label" in item["original_item"]:
            print("正在处理的 item 的所有键:", item.keys())
            questions.append(item["original_item"]["question"])
            gts.append(item["original_item"]["label"])
            extractions.append(item["Extracted answer"])
    return questions, gts, extractions


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

        with open(json_path, 'r') as f:
            data = json.load(f)#[:5]

        # data = []
        # with open(json_path, "r", encoding="utf-8") as f:
        #     for line in f:
        #         data.append(json.loads(line.strip()))

        # 获取对应模型的推理函数
        questions, ground_truths, extractions = get_multi_modal_input(args, data)
        llm, prompts, stop_token_ids = model_example_map[model](questions, ground_truths, extractions,args.model_path)

        sampling_params = SamplingParams(temperature=0,max_tokens=1024,stop_token_ids=stop_token_ids)
        
        all_results = []

        for idx in tqdm(range(len(prompts))):
            batch_inputs = [{"prompt": prompts[idx]}]
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
            for o in outputs:
                generated_text = o.outputs[0].text
                all_results.append(generated_text)
                print(f"处理进度: {len(all_results)}/{len(prompts)}")
                print(f"生成结果: {generated_text}")
                data[idx]['judge'] = generated_text
        
        result_file = args.result_file
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=4)
            
    def calculate_accuracy(result_file):
        # 读取JSON文件
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data)
        exact_match_count = 0
        
        for inst in data:
            if "judge" in inst:
                judge = inst["judge"]
                if judge.startswith("0") or judge.startswith("1"):
                    judgement = judge[0]
                elif judge.startswith("Judgement:"):
                    judgement = judge.split("Judgement:")[-1].strip()[0]
                else:
                    match = re.search(r'Judgement:\s*([0-9]+(?:\.[0-9]+)?)', judge, re.IGNORECASE)
                    if match:
                        judge = match.group(1)
                    else:
                        judgement = judge
                        print(judge)
                if judgement == "1":
                    exact_match_count += 1
        # 计算正确率
        exact_accuracy = exact_match_count / total * 100
        print(exact_match_count, "/", total)
        print("总正确率", exact_accuracy)

    calculate_accuracy(result_file)


if __name__ == "__main__":
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
                        help='Modality of the input data.')
    parser.add_argument('--json-path',
                        type=str,
                        default = f"",
                        help='Path to the JSON file containing questions and images.')

    parser.add_argument('--result-file',
                        type=str,
                        default="",
                        help='Path to save the extraction results.')
    parser.add_argument("--model-path", type=str,
                        required=True)
    args = parser.parse_args()

    main(args)

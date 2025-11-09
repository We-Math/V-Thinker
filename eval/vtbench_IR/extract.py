
import os
import re
import json
from collections import defaultdict
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question and provide the final answer at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]




def build_prompts(questions, ground_truths, responses):
    prompts = []
    for q, gt, resp in zip(questions, ground_truths, responses):
        task_description = """
        Please read the following example.
        Then extract the answer from the model response and type it at the end of the prompt.\n
        """
        question = q
        prediction = resp
        prompt = task_description
        examples = get_gpt4_ICE()
        for example in examples:
            prompt += example + '\n'
        prompt += question + '\n'
        prompt += 'Model respone: ' + prediction
        prompt += 'Extracted answer:'
        prompts.append(prompt)
    return prompts
    
DEMO_PROMPT_SCORE = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0. Output ONLY 0 or 1, do not explain.

[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\\nChoices:\\nA:2\\nB:2\\u221a{{3}}\\nC:\\u221a{{3}}\\nD:2\\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\\u221a{{3}}
Judgement: 0

[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0

[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\\nChoices:\\nA:2\\nB:2\\u221a{{3}}\\nC:\\u221a{{3}}\\nD:2\\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0

[Question]: Given the graph of the ellipse that intersects with x-axis at 9 and -9 and with y-axis at 3 and -3, determine its equation.A. \\\\frac{{x^2}}{{81}} + \\\\frac{{y^2}}{{9}} = 1 B. Can not determine.\\n
[Standard Answer]: A
[Model_answer] : \\\\frac{{x^2}}{{81}} + \\\\frac{{y^2}}{{9}} = 1
Judgement: 1

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer] : {model_answer}
Judgement: """


def build_judge_prompts(questions, ground_truths, responses):
    prompts = []
    for q, gt, resp in zip(questions, ground_truths, responses):
        full = DEMO_PROMPT_SCORE.format(question=q, gt=gt, model_answer=resp)
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{full}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)
    return prompts
def load_items(json_path: str):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        pass
    items = []
    with open(json_path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            try:
                items.append(json.loads(s))
            except Exception as e:
                print(f"[WARN] JSONL parse fail at line {ln}: {e} | snippet={s[:120]}")
    return items


def main(args):
    items = load_items(args.input_path)
    if not items:
        print("[WARN] empty input.")
        return


    if not args.reuse_judge:
        questions = [str(it.get("question_reasoning", "")) for it in items]
        gts       = [str(it.get("answer_reasoning", "")) for it in items]
        responses = [str(it.get("model_response", "")) for it in items]
        

        llm = LLM(
            model=args.model_path,
            max_model_len=4096 * 8,
            max_num_seqs=1,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        prompts  = build_prompts(questions, gts, responses)
        sampling = SamplingParams(temperature=0, max_tokens=256, stop_token_ids=None)

        for i in tqdm(range(len(prompts))):
            out = llm.generate([{"prompt": prompts[i]}], sampling_params=sampling)
            text = out[0].outputs[0].text
            items[i]["Extracted answer"] = text.strip()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Judge response1 vs answer1 using vLLM; report overall, per-domain, 2D table (domain x dataset-entity), and multi-dim stats."
    )
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-path", type=str,
                        required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8,
                        help="vLLM tensor parallel size")
    parser.add_argument("--reuse-judge", action="store_true")

    args = parser.parse_args()
    main(args)

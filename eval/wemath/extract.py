import os
import re
import json
from collections import defaultdict
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

import pandas as pd
from collections import defaultdict
import logging

def is_equal(asw: str, gt_asw: str) -> bool:
    from latex2sympy2 import latex2sympy
    if not isinstance(asw, str) or not isinstance(gt_asw, str):
        print(f'Warning: input is not string. asw: {asw}, gt_asw: {gt_asw}')
        asw, gt_asw = str(asw), str(gt_asw)

    asw = asw.lower().strip()
    gt_asw = gt_asw.lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
    except:
        pass
    return False

def list_to_dict(lst):
    try:
        actual_list = eval(lst)
        return {chr(65 + i): val for i, val in enumerate(actual_list)}
    except:
        return {}
        
def can_infer(response, choices):
    response = str(response).strip().upper()
    if not response:
        return None
        
    if response in choices:
        return response

    for opt in choices:
        if f'({opt})' in response or f'{opt}.' in response:
            return opt
    return None

def post_check(line):
    res = None
    ans = line.get('answer', '')
    response = line.get('Extracted answer', '')
    
    try:
        choices_str = line.get('choices')
        if choices_str and len(eval(choices_str)) > 0:
            choices_dict = list_to_dict(choices_str)
            ans = line.get('answer', '') 
            inferred_choice = can_infer(response, choices_dict)
            if inferred_choice and inferred_choice == ans:
                return True
            if inferred_choice and is_equal(choices_dict.get(inferred_choice, ''), ans):
                 return True
            res = inferred_choice if inferred_choice else response
        else:
            res = str(response)
            ans = str(ans)
    except Exception:
        res = str(response)
        ans = str(ans)

    try:
        return is_equal(res, ans)
    except Exception as err:
        logging.warning(f'{type(err)}: {err}')
        return False


def summarize_with_post_check(items, category_key="domain"):
    """
    """
    print("\n== Accuracy calculated by post_check (MATH_V_acc logic) ==")
    
    tot = defaultdict(int)
    hit = defaultdict(int)
    
    for item in tqdm(items, desc="Calculating accuracy with post_check"):
        cate = item.get(category_key, "Overall")
        tot["Overall"] += 1
        if cate != "Overall":
            tot[cate] += 1
        
        if post_check(item):
            hit["Overall"] += 1
            if cate != "Overall":
                hit[cate] += 1

    res = defaultdict(list)
    if "Overall" in tot:
        k = "Overall"
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] > 0 else 0)
    sorted_keys = sorted([k for k in tot.keys() if k != "Overall"])
    for k in sorted_keys:
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100 if tot[k] > 0 else 0)

    df = pd.DataFrame(res)
    print(df.to_string(index=False))
    
    return df


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


def build_mathv_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


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


def to_zero_or_one(text) -> str:
    if text is None:
        return "0"
    s = str(text).strip()
    for ch in s:
        if ch == "0":
            return "0"
        if ch == "1":
            return "1"
    return "0"

def summarize_overall_and_domain(items):
    total = len(items)
    correct = sum(1 for it in items if str(it.get("judge", "0")) == "1")
    overall_acc = (correct / total * 100) if total else 0.0

    domain_tot = defaultdict(int)
    domain_cor = defaultdict(int)
    for it in items:
        dom = it.get("domain", "unknown")
        domain_tot[dom] += 1
        if str(it.get("judge", "0")) == "1":
            domain_cor[dom] += 1

    print("\n== Overall ==")
    print(f"{correct}/{total}  Acc={overall_acc:.2f}%")

    print("\n== By domain ==")
    for d in sorted(domain_tot.keys()):
        cor, tot = domain_cor[d], domain_tot[d]
        acc = (cor / tot * 100) if tot else 0.0
        print(f"{d}: {cor}/{tot} = {acc:.2f}%")

    domain_stats = {d: {"correct": domain_cor[d], "total": domain_tot[d],
                        "acc": (domain_cor[d]/domain_tot[d]*100) if domain_tot[d] else 0.0}
                    for d in domain_tot.keys()}
    return overall_acc, domain_stats

def summarize_2d(items, row_key="domain", col_key="dataset-entity", show_acc_only=False):
    proc = []
    for it in items:
        j = 1 if str(it.get("judge", "0")) == "1" else 0
        r = str(it.get(row_key, "<none>"))
        c = str(it.get(col_key, "<none>"))
        proc.append((r, c, j))

    row_set = sorted(set(r for r, _, _ in proc))
    col_set = sorted(set(c for _, c, _ in proc))

    cell = defaultdict(lambda: {"cor": 0, "tot": 0})
    row_tot = defaultdict(lambda: {"cor": 0, "tot": 0})
    col_tot = defaultdict(lambda: {"cor": 0, "tot": 0})
    overall_cor = overall_tot = 0

    for r, c, j in proc:
        cell[(r, c)]["tot"] += 1
        cell[(r, c)]["cor"] += j
        row_tot[r]["tot"] += 1
        row_tot[r]["cor"] += j
        col_tot[c]["tot"] += 1
        col_tot[c]["cor"] += j
        overall_tot += 1
        overall_cor += j

    header = [f"{row_key} \\ {col_key}"] + col_set + ["Row Total"]
    print("\n== 2D table by", row_key, "x", col_key, "==")
    print(" | ".join(header))
    for r in row_set:
        row_out = [r]
        for c in col_set:
            stat = cell[(r, c)]
            cor, tot = stat["cor"], stat["tot"]
            acc = (cor / tot * 100) if tot else 0.0
            row_out.append(f"{acc:.2f}%" if show_acc_only else f"{cor}/{tot} ({acc:.2f}%)")
        rc, rt = row_tot[r]["cor"], row_tot[r]["tot"]
        racc = (rc / rt * 100) if rt else 0.0
        row_out.append(f"{racc:.2f}%" if show_acc_only else f"{rc}/{rt} ({racc:.2f}%)")
        print(" | ".join(row_out))

    col_line = ["Col Total"]
    for c in col_set:
        cc, ct = col_tot[c]["cor"], col_tot[c]["tot"]
        cacc = (cc / ct * 100) if ct else 0.0
        col_line.append(f"{cacc:.2f}%" if show_acc_only else f"{cc}/{ct} ({cacc:.2f}%)")
    oacc = (overall_cor / overall_tot * 100) if overall_tot else 0.0
    col_line.append(f"{oacc:.2f}%" if show_acc_only else f"{overall_cor}/{overall_tot} ({oacc:.2f}%)")
    print(" | ".join(col_line))

    table = {
        "rows": row_set, "cols": col_set,
        "cells": {f"{r}|||{c}": cell[(r, c)] for r in row_set for c in col_set},
        "row_totals": {r: row_tot[r] for r in row_set},
        "col_totals": {c: col_tot[c] for c in col_set},
        "overall": {"correct": overall_cor, "total": overall_tot, "acc": oacc},
        "row_key": row_key, "col_key": col_key,
        "format": "acc_only" if show_acc_only else "frac+acc",
    }
    return table

def export_csv(table, out_csv, show_acc_only=False):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows, cols, cells = table["rows"], table["cols"], table["cells"]
    row_totals = table["row_totals"]
    with open(out_csv, "w", encoding="utf-8") as f:
        header = [f"{table['row_key']} \\ {table['col_key']}"] + cols + ["Row Total"]
        f.write(",".join(header) + "\n")
        for r in rows:
            line = [r]
            for c in cols:
                st = cells[f"{r}|||{c}"]
                cor, tot = st["cor"], st["tot"]
                acc = (cor / tot * 100) if tot else 0.0
                line.append(f"{acc:.2f}%" if show_acc_only else f"{cor}/{tot} ({acc:.2f}%)")
            rc, rt = row_totals[r]["cor"], row_totals[r]["tot"]
            racc = (rc / rt * 100) if rt else 0.0
            line.append(f"{racc:.2f}%" if show_acc_only else f"{rc}/{rt} ({racc:.2f}%)")
            f.write(",".join(line) + "\n")
    print(f"[OK] CSV saved -> {out_csv}")

def export_json(table, out_json):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON saved -> {out_json}")

def _extract_field_values(item, key):
    v = item.get(key, None)
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, (list, dict)):
                out.append(json.dumps(x, ensure_ascii=False))
            else:
                out.append(str(x).strip())
        return [s for s in out if s]
    s = str(v).strip()
    if not s:
        return []
    if "," in s or "，" in s or ";" in s or "；" in s:
        for sep in [",", "，", ";", "；"]:
            s = s.replace(sep, ",")
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _derive_features(it):
    q = str(it.get("question1", "")).lower()
    gt = str(it.get("answer1", ""))
    feats = {}
    feats["has_figure"] = "yes" if any(tok in q for tok in ["figure", "下图", "如图", "see figure", "shown in the figure"]) else "no"
    qlen = len(str(it.get("question1", "")))
    feats["q_len_bin"] = "<80" if qlen < 80 else "80-199" if qlen < 200 else "200-399" if qlen < 400 else ">=400"
    if re.search(r"^[A-D]\s*$", gt.strip(), re.I):
        feats["ans_type"] = "mcq"
    elif re.search(r"\d", gt) and len(gt) <= 16:
        feats["ans_type"] = "numeric/short"
    else:
        feats["ans_type"] = "other"
    return feats

def summarize_multidim(items, group_by=None, explode_fields=None, extra_derived=True, topk_preview=50):
    group_by = group_by or []
    explode_fields = set(explode_fields or [])

    proc = []
    for it in items:
        judge = 1 if str(it.get("judge", "0")) == "1" else 0
        base = dict(it)
        base["_judge"] = judge
        if extra_derived:
            base.update(_derive_features(it))
        proc.append(base)

    total = len(proc)
    correct = sum(x["_judge"] for x in proc)
    overall_acc = (correct / total * 100) if total else 0.0

    if not group_by:
        print("\n== Overall ==")
        print(f"{correct}/{total}  Acc={overall_acc:.2f}%")
        return {"overall": {"correct": correct, "total": total, "acc": overall_acc}, "groups": {}}

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for it in proc:
        lists = []
        for key in group_by:
            if key in explode_fields:
                vals = _extract_field_values(it, key)
                vals = vals or ["<none>"]
                lists.append(vals)
            else:
                v = it.get(key, "<none>")
                lists.append([str(v)])

        def dfs(i, cur):
            if i == len(lists):
                tup = tuple(cur)
                stats[tup]["total"] += 1
                stats[tup]["correct"] += it["_judge"]
                return
            for v in lists[i]:
                dfs(i+1, cur + [v])

        dfs(0, [])

    print("\n== By " + " x ".join(group_by) + " ==")
    rows = []
    for k, v in stats.items():
        acc = (v["correct"] / v["total"] * 100) if v["total"] else 0.0
        rows.append((k, v["correct"], v["total"], acc))
    rows.sort(key=lambda r: (tuple(str(x) for x in r[0]), -r[3], -r[1], r[2]))
    for k, c, t, a in rows[:max(1, topk_preview)]:
        print(f"{k}: {c}/{t} = {a:.2f}%")
    if len(rows) > topk_preview:
        print(f"... ({len(rows)-topk_preview} more groups)")

    return {
        "overall": {"correct": correct, "total": total, "acc": overall_acc},
        "groups": {
            "keys": group_by,
            "explode_fields": list(explode_fields),
            "items": [{"key": list(k), "correct": c, "total": t, "acc": a} for (k, c, t, a) in rows],
        },
    }


def main(args):
    items = load_items(args.input_path)
    if not items:
        print("[WARN] empty input.")
        return

    if not args.reuse_judge:
        questions = [str(it.get("question", "")) for it in items]
        gts       = [str(it.get("answer", "")) for it in items]
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
    print(f"[OK] details saved -> {args.output_path}")

    overall_acc, domain_stats = summarize_overall_and_domain(items)

    table_2d = summarize_2d(
        items,
        row_key=args.row_key,
        col_key=args.col_key,
        show_acc_only=args.acc_only
    )
    if args.csv_path:
        export_csv(table_2d, args.csv_path, show_acc_only=args.acc_only)
    if args.json_table_path:
        export_json(table_2d, args.json_table_path)

    group_by = [s.strip() for s in (args.group_by or "").split(",") if s.strip()]
    explode_fields = [s.strip() for s in (args.explode_fields or "").split(",") if s.strip()]
    multi_metrics = summarize_multidim(
        items,
        group_by=group_by,
        explode_fields=explode_fields,
        extra_derived=True,
        topk_preview=args.topk_preview
    )

    if args.metrics_path:
        metrics = {
            "overall": overall_acc,
            "by_domain": domain_stats,
            "table_2d": table_2d,
            "multidim": multi_metrics,
        }
        os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
        with open(args.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[OK] metrics saved -> {args.metrics_path}")

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Judge response1 vs answer1 using vLLM; report overall, per-domain, 2D table (domain x dataset-entity), and multi-dim stats."
    )
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--metrics-path", type=str, default="")

    parser.add_argument("--model-path", type=str,
                        required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8,
                        help="vLLM tensor parallel size")
    parser.add_argument("--reuse-judge", action="store_true")

    parser.add_argument("--row-key", type=str, default="domain")
    parser.add_argument("--col-key", type=str, default="dataset-entity")
    parser.add_argument("--acc-only", action="store_true")
    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--json-table-path", type=str, default="")
    parser.add_argument("--row-keep", type=str, default="")
    parser.add_argument("--col-keep", type=str, default="")

    parser.add_argument("--group-by", type=str, default="")
    parser.add_argument("--explode-fields", type=str, default="")
    parser.add_argument("--topk-preview", type=int, default=50)

    args = parser.parse_args()
    main(args)

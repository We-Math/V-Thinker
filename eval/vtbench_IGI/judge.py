# -*- coding: utf-8 -*-
# judge_visual_tasks_english_prompt.py

import os
import re
import json
import base64
from io import BytesIO
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

# -----------------------------
# English Version: Final Judging Prompt with Instruction, Original, Generated, and GT Images
# -----------------------------
ENGLISH_FINAL_PROMPT_IMAGE_JUDGE = """ Task Description: You are an expert visual evaluator. Your task is to determine if the auxiliary lines or modifications in the [Generated Image] are conceptually correct according to the [Instruction], using the [Ground Truth Image] as a reference for the correct concept.

Judging Criteria: The primary goal is to assess if the geometric concept is correctly applied, not to enforce pixel-perfect accuracy.

Consistent (Judgement: 1): The generated image is considered consistent if it meets the following conditions:

Conceptual Correctness: The modifications correctly follow the geometric/logical concept of the instruction. The key is that the auxiliary lines connect the correct intended features (e.g., connecting the correct vertices, bisecting the correct angle, drawing a perpendicular from the right point).

Tolerance for Imperfection: The result is visually and conceptually similar to the [Ground Truth Image]. Minor inaccuracies in position, angle, or length are acceptable as long as the core intent is preserved.

Irrelevant Differences: Differences in color, line thickness, or line style (e.g., dashed vs. solid) should be ignored.

Inconsistent (Judgement: 0): The generated image is considered inconsistent if:

Conceptual Error: The modifications are conceptually wrong (e.g., connecting the wrong vertices, bisecting the wrong angle, dropping a perpendicular to the wrong line).

Missing or Incomplete: The required modifications are missing, incomplete, or do not follow the core instruction.

No Change: The [Generated Image] is identical to the [Original Image] (no effective modifications were made).

Output Format:

If consistent, output 1.

If inconsistent, output 0.

Output ONLY 0 or 1. Do not provide any explanation.

[Task]

[Instruction]: {instruction}

Please refer to the following three images to make your judgement:

[Original Image] (The image before the instruction was applied) <|vision_start|><|image_pad|><|vision_end|>

[Generated Image] (The image produced by the model to be evaluated) <|vision_start|><|image_pad|><|vision_end|>

[Ground Truth Image] (The correct reference image) <|vision_start|><|image_pad|><|vision_end|>

Please compare all the information above and provide your judgement. Judgement:"""



# (其他函数 load_items, to_zero_or_one, encode_image_to_base64 保持不变)
def load_items(json_path: str):
    with open(json_path, "r", encoding="utf-8-sig") as f: raw = f.read().strip()
    if not raw: return []
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except Exception: pass
    items = []
    with open(json_path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"): continue
            try: items.append(json.loads(s))
            except Exception as e: print(f"[WARN] JSONL parse fail at line {ln}: {e} | snippet={s[:120]}")
    return items

def to_zero_or_one(text) -> str:
    if text is None: return "0"
    s = str(text).strip()
    for ch in s:
        if ch == "0": return "0"
        if ch == "1": return "1"
    return "0"

def encode_image_to_base64(image_path: str):
    try:
        with Image.open(image_path) as img:
            img.thumbnail((1024, 1024))
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"[WARN] Failed to load or encode image at {image_path}: {e}")
        return None

def build_multimodal_prompts(items):
    prompts_data = []
    for it in items:
        instruction = it.get("original_item", {}).get("instruction_guided_interaction", "")
        orig_img_path = it.get("original_item", {}).get("image_path", "")
        gen_img_path = it.get("aux_path", "")
        gt_img_path = it.get("original_item", {}).get("instruction_guided_interaction_answer_path", "")
        print(gt_img_path)
        if not instruction:
            prompts_data.append({"prompt": "Cannot judge: missing instruction (question2)", "multi_modal_data": None, "valid": False})
            print("wo prompt")
            #print("{"prompt": "Cannot judge: missing instruction (question2)", "multi_modal_data": None, "valid": False}")
            continue
        if not (orig_img_path and os.path.exists(orig_img_path)):
            print("1")
        if not (gen_img_path and os.path.exists(gen_img_path)):
            print("2")
        if not (gt_img_path and gt_img_path != "null" and os.path.exists(gt_img_path)):
            print("3")       
        if not all([
            orig_img_path and os.path.exists(orig_img_path),
            gen_img_path and os.path.exists(gen_img_path),
            gt_img_path and gt_img_path != "null" and os.path.exists(gt_img_path)
        ]):
            prompts_data.append({"prompt": "Cannot judge: one or more image paths are invalid", "multi_modal_data": None, "valid": False})
            print("wo image")
            #print("{"prompt": "Cannot judge: one or more image paths are invalid", "multi_modal_data": None, "valid": False}")
            continue
            # -----------------------------
        try:
            orig_img_pil = Image.open(it.get("original_item", {}).get("image_path", "")).convert("RGB")
            gen_img_pil = Image.open(it.get("aux_path", "")).convert("RGB")
            gt_img_pil = Image.open(it.get("original_item", {}).get("instruction_guided_interaction_answer_path", "")).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load one of the images as PIL object: {e}")
            prompts_data.append({"prompt": "Cannot judge: image loading failed", "pil_images": None, "valid": False})
            continue
            
        text_prompt = ENGLISH_FINAL_PROMPT_IMAGE_JUDGE.format(
            instruction=instruction
        )
        prompt_for_vllm = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{text_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        pil_images_for_vllm = [orig_img_pil, gen_img_pil, gt_img_pil]

        prompts_data.append({
            "prompt": prompt_for_vllm, 
            "pil_images": pil_images_for_vllm, 
            "valid": True
        })

    return prompts_data


# -----------------------------
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

    if args.row_keep or args.col_keep:
        row_keep = set([s.strip() for s in args.row_keep.split(",") if s.strip()])
        col_keep = set([s.strip() for s in args.col_keep.split(",") if s.strip()])
        filt = []
        for it in items:
            r = str(it.get(args.row_key, "<none>"))
            c = str(it.get(args.col_key, "<none>"))
            if (not row_keep or r in row_keep) and (not col_keep or c in col_keep):
                filt.append(it)
        items = filt
        print(f"[INFO] after filter: {len(items)} items")
        
    if not args.reuse_judge:
        llm = LLM(
            model=args.model_path,
            max_num_seqs=1,
            max_model_len=196608,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        
        prompts_data = build_multimodal_prompts(items)
        sampling = SamplingParams(temperature=0, max_tokens=256, stop_token_ids=None)

        for i, p_data in enumerate(tqdm(prompts_data, desc="Judging with VLM")):
            if not p_data["valid"]:
                items[i]["judge_raw"] = p_data["prompt"]
                items[i]["judge"] = "0"
                continue
            batch_inputs = [{"prompt": p_data["prompt"], "multi_modal_data": { "image": p_data["pil_images"] }}] 
            out = llm.generate(
                batch_inputs, 
                sampling_params=sampling
            )
            text = out[0].outputs[0].text
            items[i]["judge_raw"] = text
            items[i]["judge"] = to_zero_or_one(text)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Judgement details saved -> {args.output_path}")
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

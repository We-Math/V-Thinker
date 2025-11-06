# parallel_inference.py

import torch
import os
import json
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration
from tqdm import tqdm
from utils import run_evaluation # Assuming you have this utility function
MODEL_PATH=""

config = AutoConfig.from_pretrained(MODEL_PATH)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto", # "auto" works perfectly with CUDA_VISIBLE_DEVICES
    config=config
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

question_text = "Question: Hint: Please answer the question and provide the final answer at the end.\nQuestion: How many lines of symmetry does this figure have?\n\n\nPlease provide the final answer in the format <answer>X</answer>"
image_path = "./224.png"
    
    # Construct the full, normalized image pat
final_assistant_response, final_answer, aux_path = run_evaluation(question_text, image_path, "./", model, processor)
print("Model Response")
print(final_answer)
print("auxiliary path")
print(final_answer)   

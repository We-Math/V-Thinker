#!/bin/bash
NUM_GPUS=8
EVAL_NAME="vthinker" 
BASE_DIR="./results"
BENCH_NAME="vtoolbench"
PARALLEL_RESULTS_DIR="./parallel_results_${EVAL_NAME}"
MERGED_OUTPUT_FILE="$BASE_DIR/${EVAL_NAME}_${BENCH_NAME}_inf.json"
EXTRACT_OUTPUT_FILE="$BASE_DIR/${EVAL_NAME}_${BENCH_NAME}_ext.json"
JUDGE_OUTPUT_FILE="$BASE_DIR/${EVAL_NAME}_${BENCH_NAME}_judge.json"
DATA_PATH="../../data/Vtbench.json"
MODEL_PATH="../../model/"
AUX_PATH="./vthinker_aux"
JUDGE_MODEL="Qwen--Qwen3-VL-235B-A22B-Instruct"
mkdir -p $PARALLEL_RESULTS_DIR
mkdir -p $(dirname $MERGED_OUTPUT_FILE)


for i in $(seq 0 $(($NUM_GPUS - 1)))
do
  echo "Starting worker $i on GPU $i..."
  
  CUDA_VISIBLE_DEVICES=$i python parallel_inference_vthink.py \
    --gpu_id $i \
    --worker_id $i \
    --num_workers $NUM_GPUS \
    --output_dir $PARALLEL_RESULTS_DIR \
    --model_path $MODEL_PATH \
    --data_json_path $DATA_PATH \
    --aux_outpath $AUX_PATH &
done

wait



python merge_results.py \
    --results-dir "$PARALLEL_RESULTS_DIR" \
    --output-file "$MERGED_OUTPUT_FILE"




python judge.py \
    --input-path "$MERGED_OUTPUT_FILE" \
    --model-path "$JUDGE_MODEL"\
    --output-path "$JUDGE_OUTPUT_FILE"
    
export NPROC_PER_NODE=8
export WORLD_SIZE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond1 
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL=0

nnodes=16
nproc_per_node=8
NPROC_PER_NODE=$nproc_per_node \

MASTER_PORT=23456 \
MAX_PIXELS=4194304 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=$nnodes \
NODE_RANK=$RANK \
MASTER_ADDR=29.163.176.51 \
MASTER_PORT=29500 \
swift sft \
    --model_type qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --num_train_epochs 1 \
    --train_type full \
    --deepspeed zero2 \
    --tuner_backend peft \
    --torch_dtype bfloat16 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --learning_rate 1e-5 \
    --eval_steps 10240 \
    --save_steps 300 \
    --max_length 15000 \
    --output_dir checkpoints_perception \
    --dataset '../data/perception.jsonl' \
    --dataloader_num_workers 4 \
    --per_device_train_batch_size 8

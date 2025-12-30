#!/bin/bash

# Runs the "340M" parameter model with Distributed Muon
# See more details at: https://github.com/MoonshotAI/Moonlight/blob/master/Moonlight.pdf
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_BASE_URL=https://api.bandw.top
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY="d6abca388ed2d010b2b2d3057401ad0db45f12ee" # 请替换为您的 WandB API Key
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

C4_HOME=/ssd/hrz/datasets/c4_llama2tokenizer
DATA_BLEND=""
for i in {00000..00499}; do # 1/25
	DATA_BLEND="${DATA_BLEND} 0.002 ${C4_HOME}/c4_llama2_${i}_text_document"
done

DATA_PATH="$DATA_BLEND"

CHECKPOINT_PATH=./muon_baseline_ckp
TENSORBOARD_LOGS_PATH=./muon_baseline_ckp


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 24
    --hidden-size 768
    --ffn-hidden-size 2560
    --num-attention-heads 16
    --seq-length 1024
    --max-position-embeddings 1024 
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --transformer-impl local
    --use-rotary-position-embeddings
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --rotary-base 1000000 
    --swiglu
    --use-flash-attn
)

TRAINING_ARGS=(
    --optimizer muon
    --micro-batch-size 64 
    --global-batch-size 1024 
    --train-iters 5000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.02 
    --clip-grad 1.0 
    --bf16
    --lr 3e-3
    --lr-decay-style cosine 
    --min-lr 3e-4
    --muon-matched-adamw-rms 0.2
    --lr-warmup-fraction 0.02
    --lr-decay-iters 5000
    --use-distributed-optimizer
    --ckpt-format torch
    
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model /ssd/hrz/megatron_optimizer/Megatron-LM/llama2tokenizer/tokenizer.model
    --data-path $DATA_PATH
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 20
    --save-interval 10000 
    --eval-interval 500 
    --save $CHECKPOINT_PATH 
    --eval-iters 50
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --wandb-project ${WANDB_PROJECT:-"muon_megatron"}
    --wandb-exp-name ${WANDB_NAME:-"muon_baseline"}
)

torchrun ${DISTRIBUTED_ARGS[@]} /ssd/hrz/megatron_optimizer/Megatron-LM/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

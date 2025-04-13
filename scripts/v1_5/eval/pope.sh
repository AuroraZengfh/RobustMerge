#!/bin/bash

if [ ! -n "$1" ] ;then
    MODELPATH='/mnt/ShareDB_6TB/datasets/MLLM_CL/checkpoint/mergeing/r16/ScienceQA_llava_lora_epoch_1'
else
    MODELPATH=$1
fi

if [ ! -n "$2" ] ;then
    GPU=0
else
    GPU=$2
fi
RESULT_DIR="./results/pope"
echo $RESULT_DIR

CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_vqa_loader \
    --model-path $MODELPATH \
    --model-base /mnt/ShareDB_6TB/models/llava-v1.5-7b \
    --question-file /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file $RESULT_DIR/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/pope/coco \
    --question-file /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $RESULT_DIR/llava-v1.5-7b.jsonl

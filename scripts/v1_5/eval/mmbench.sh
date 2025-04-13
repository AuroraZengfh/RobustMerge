#!/bin/bash

SPLIT="mmbench_dev_20230712"

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

MODEL_TYPE=$3

RESULT_DIR="./results/mmbench"
echo $RESULT_DIR

CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_vqa_mmbench \
    --model-path $MODELPATH \
    --model-base /mnt/ShareDB_6TB/models/llava-v1.5-7b \
    --question-file /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file $RESULT_DIR/$MODEL_TYPE/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p $RESULT_DIR/$MODEL_TYPE/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir $RESULT_DIR/$MODEL_TYPE \
    --upload-dir $RESULT_DIR/$MODEL_TYPE/$SPLIT \
    --experiment llava-v1.5-7b

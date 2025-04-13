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

CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_vqa_loader \
    --model-path $MODELPATH \
    --model-base /mnt/ShareDB_6TB/models/llava-v1.5-7b \
    --question-file /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/MME/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /mnt/ShareDB_6TB/datasets/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b

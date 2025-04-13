#!/bin/bash

CHUNKS=1
IDX=0

MODELPATH=$1
GPU=$2

RESULT_DIR="./results/ScienceQA"
echo $RESULT_DIR

CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_vqa_science \
    --model-path $MODELPATH \
    --model-base models/llava-v1.5-7b \
    --question-file instructions/ScienceQA/test.json \
    --image-folder datasets \
    --answers-file $RESULT_DIR/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &

wait

output_file=$RESULT_DIR/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_science_qa.py \
    --base-dir  datasets/ScienceQA/ \
    --result-file $output_file \
    --output-file $RESULT_DIR/output.jsonl \
    --output-result $RESULT_DIR/output_result.jsonl \
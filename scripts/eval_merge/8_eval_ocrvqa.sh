#!/bin/bash

CHUNKS=1
IDX=0

MODELPATH=$1
GPU=$2

RESULT_DIR="./results/OCRVQA"
echo $RESULT_DIR

CUDA_VISIBLE_DEVICES=$GPU python -m llava.eval.model_ocr_vqa \
    --model-path $MODELPATH \
    --model-base models/llava-v1.5-7b \
    --question-file instructions/OCRVQA/test.json \
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

python llava/eval/eval_ocrvqa.py \
    --annotation-file instructions/OCRVQA/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR \
CUDA_VISIBLE_DEVICES=0 python scripts/merge/merge_lora_weights.py --model-path /path/to/your-fined-model \
    --model-base models/llava-v1.5-7b --save-model-path /path/to/yout/merged/checkpoint
# zero shot
# sh scripts/v1_5/eval/pope.sh /mnt/ShareDB_6TB/models/llava-v1.5-7b-lora 0

# MTL
# sh scripts/eval_merge/1_eval_scienceqa.sh /mnt/ShareDB_6TB/datasets/MLLM_CL/checkpoint/mergeing/r16/mtl_llava_lora_epoch_1 0

# ------------------------------------------------------ #### ------------------------------------------------------------------------

# each method
# eval_path=LLaVA_lora_merge_ours

# sh scripts/v1_5/eval/pope.sh checkpoints/${eval_path} 0 
# sh scripts/v1_5/eval/mme.sh checkpoints/${eval_path} 0
# sh scripts/v1_5/eval/mmbench.sh checkpoints/${eval_path} 0 ties



# eval_path=LLaVA_lora_merge_reimple_dare

# sh scripts/v1_5/eval/pope.sh checkpoints/${eval_path} 1 

eval_path=LLaVA_lora_merge_ours_1.2

# sh scripts/v1_5/eval/mmbench.sh checkpoints/${eval_path} 0 ours
sh scripts/v1_5/eval/pope.sh checkpoints/${eval_path} 1 

# eval_path=LLaVA_lora_merge_reimple_ta
# sh scripts/v1_5/eval/mmbench.sh checkpoints/${eval_path} 0

# eval_path=LLaVA_lora_merge_reimple_ties

# sh scripts/v1_5/eval/mmbench.sh checkpoints/${eval_path} 0 ties

# eval_path=LLaVA_lora_merge_reimple_pcb

# sh scripts/v1_5/eval/mmbench.sh checkpoints/${eval_path} 0 pcb

# sh scripts/v1_5/eval/mmbench.sh /mnt/ShareDB_6TB/datasets/MLLM_CL/checkpoint/mergeing/r16/mtl_llava_lora_epoch_1 0 mtl

# eval_path=LLaVA_lora_merge_ours
# sh scripts/v1_5/eval/mmbench.sh checkpoints/${eval_path} 0 ours

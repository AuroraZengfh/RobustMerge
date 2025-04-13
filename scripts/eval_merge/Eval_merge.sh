eval_path=/path/to/yout/merged/checkpoint

sh scripts/eval_merge/1_eval_scienceqa.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/2_eval_vizwiz_caption.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/3_eval_ImageNet.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/4_eval_vqav2.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/5_eval_Iconqa.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/6_eval_flickr30k.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/7_eval_grounding.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/8_eval_ocrvqa.sh checkpoints/${eval_path} 0


sh scripts/eval_merge/eval_aokvqa.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/eval_imagenetr.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/eval_screen2words.sh checkpoints/${eval_path} 0
sh scripts/eval_merge/eval_tabmwp.sh checkpoints/${eval_path} 0

# deepspeed --num_gpus 1 main_ood_param_check.py --overwrite_output_dir --split --task_name banking77 --model_name_or_path gpt2-xl --ds_config ds_configs_samples/zero1_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_lora --lora_r 25
deepspeed --include localhost:2 --master_port 29499 main_ood_param_check.py --overwrite_output_dir --split --task_name banking77 --model_name_or_path gpt2-medium --ds_config ds_configs_samples/zero1_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_adapter --adapter_size 4

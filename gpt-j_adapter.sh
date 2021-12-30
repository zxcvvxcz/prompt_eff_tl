deepspeed --num_gpus 1 main_ood_param_check.py --task_name clinc150 --overwrite_output_dir --model_name_or_path EleutherAI/gpt-j-6B --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/clinc150/full/gpt-j-6B --num_train_epochs 40 --seed 1234 --cache_dir ~/data/model_data --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --apply_adapter --adapter_size 62

# deepspeed main_ood_param_check.py --split --task_name banking77 --model_name_or_path EleutherAI/gpt-j-6B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/banking77/0.5/gpt-j-6B --num_train_epochs 40 --seed 1234 --cache_dir ~/data/model_data --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400
# deepspeed main_ood_param_check.py --split --task_name clinc150 --model_name_or_path EleutherAI/gpt-j-6B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/clinc150/0.5/gpt-j-6B --num_train_epochs 40 --seed 1234 --cache_dir ~/data/model_data --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400
# deepspeed main_ood_param_check.py --split --task_name snips --model_name_or_path EleutherAI/gpt-j-6B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/snips/0.5/gpt-j-6B --num_train_epochs 40 --seed 1234 --cache_dir ~/data/model_data --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400
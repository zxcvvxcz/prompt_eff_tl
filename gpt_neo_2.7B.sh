deepspeed main_ood.py --overwrite_output_dir --task_name clinc150 --model_name_or_path EleutherAI/gpt-neo-2.7B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/clinc150/full/gpt-neo-2.7B --num_train_epochs 40 --seed 1234 --cache_dir model_data_gpt_neo_2.7B --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400

# deepspeed main_ood.py --split --task_name banking77 --model_name_or_path EleutherAI/gpt-neo-2.7B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/banking77/0.5/gpt-neo-2.7B --num_train_epochs 40 --seed 1234 --cache_dir model_data_gpt_neo_2.7B --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400
# deepspeed main_ood.py --split --task_name clinc150 --model_name_or_path EleutherAI/gpt-neo-2.7B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/clinc150/0.5/gpt-neo-2.7B --num_train_epochs 40 --seed 1234 --cache_dir model_data_gpt_neo_2.7B --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400
# deepspeed main_ood.py --split --task_name snips --model_name_or_path EleutherAI/gpt-neo-2.7B --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/snips/0.5/gpt-neo-2.7B --num_train_epochs 40 --seed 1234 --cache_dir model_data_gpt_neo_2.7B --lr 0.00016 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400
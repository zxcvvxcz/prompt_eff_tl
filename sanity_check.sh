deepspeed --include localhost:0,1 --master_port 29484 main_ood.py --overwrite_output_dir --task_name banking77 --split --split_ratio 0.25 --model_name_or_path gpt2 --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 1e-3 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim 4
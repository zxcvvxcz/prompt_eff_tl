deepspeed --num_gpus=1 main_ood.py --split --task_name banking77 --model_name_or_path gpt2-medium --ds_config ds_configs_samples/zero3_config.json --output_dir test_output --num_train_epochs 20 --seed 1234
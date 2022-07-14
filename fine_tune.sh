# deepspeed --num_gpus 1 main_ood.py --overwrite_output_dir --split --task_name clinc150 --model_name_or_path gpt2 --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_lora --lora_r 3
# deepspeed --include localhost:2 --master_port 29499 main_ood.py --overwrite_output_dir --split --task_name banking77 --model_name_or_path gpt2-medium --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_adapter --adapter_size 4
# deepspeed --include localhost:2 --master_port 29499 main_ood.py --overwrite_output_dir --split --task_name banking77 --model_name_or_path gpt2-medium --ds_config ds_configs_samples/zero3_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_adapter --adapter_size 4

master_port=29498
localhost=1,2
ds_config=zero3_config.json
for model_name in gpt2-xl
# for model_name in EleutherAI/gpt-j-6B gpt2 gpt2-medium gpt2-xl gpt2-large EleutherAI/gpt-neo-2.7B
do
    for task_name in banking77 clinc150
    do
        # for lr in 1e-4 2e-4 5e-5 1e-3
        # for lr in 2e-4 5e-5 1e-3
        for lr in 1e-3 1e-4 1e-5 5e-5
        do
            for seed in 42
            do
                echo $task_name, $model_name, $lr, $seed
                if [ $task_name = clinc150 ]
                then
                    deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/$ds_config --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --skip_ood
                    deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/$ds_config --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --skip_ood

                else
                    deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/$ds_config --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --skip_ood
                fi
            done
        done
    done
done
# lora, prefix, adapter
# for model_name in gpt2 gpt2-medium gpt2-large gpt2-xl
for model_name in gpt2 gpt2-medium gpt2-large gpt2-xl
do
    for task_name in clinc150 banking77
    do
        for lr in 1e-4 2e-4 1e-3 1e-5
        do
            echo $model_name
            echo $task_name
            echo $lr
            if [ $task_name = clinc150 ]
            then
                deepspeed --include localhost:0,1 main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_linear
                deepspeed --include localhost:0,1 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_linear
            else
                deepspeed --include localhost:0,1 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_linear
            fi
        done
    done
done


bash gpt2_fine_tune_clinc_25.sh
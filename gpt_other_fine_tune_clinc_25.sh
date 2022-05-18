
for model_name in EleutherAI/gpt-j-6B EleutherAI/gpt-neo-2.7B
do
    for task_name in clinc150
    do
        for lr in 2e-5 1e-5 5e-5
        do
            echo $model_name
            echo $task_name
            echo $lr
            deepspeed --include localhost:0,1 --master_port 29493 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_adapter --adapter_size $a
        done
    done
done
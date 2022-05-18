for task_name in clinc150
do
    for model_name in gpt2-large gpt2-medium gpt2
    do
        for lr in 2e-5 1e-5 5e-5
        do
            echo $model_name
            echo $task_name
            echo $lr
            deepspeed --include localhost:0,1 --master_port 29493 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200
        done
    done
done
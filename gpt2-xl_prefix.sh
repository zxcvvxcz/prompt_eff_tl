for model_name in gpt2-xl
do
    if [ $model_name = gpt2 ]
    then
        p_dim_list="4 31 63"
    elif [ $model_name = gpt2-medium ]
    then 
        p_dim_list="6 34 72"
    elif [ $model_name = gpt2-large ]
    then
        p_dim_list="7 40 82"
    elif [ $model_name = gpt2-xl ]
    then
        p_dim_list="9 49 100"
    fi
    for task_name in banking77
    do
        for lr in 1e-4 2e-4 1e-5
        do
            echo $model_name
            echo $task_name
            echo $lr
            for p_dim in $p_dim_list
            do
                if [ $task_name = clinc150 ]
                then
                    deepspeed --include localhost:1,2 --master_port 29497 main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                    deepspeed --include localhost:1,2 --master_port 29497 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                else
                    deepspeed --include localhost:1,2 --master_port 29497 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                fi
            done
        done
    done
done
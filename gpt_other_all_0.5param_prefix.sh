# for model_name in EleutherAI/gpt-j-6B EleutherAI/gpt-neo-2.7B
for model_name in EleutherAI/gpt-j-6B
do
    if [ $model_name = EleutherAI/gpt-j-6B ]
    then
        r_list="64"
        a_list="62"
        p_dim_list="120"
    elif [ $model_name = EleutherAI/gpt-neo-2.7B ]
    then 
        r_list="40"
        a_list="39"
        p_dim_list="76"
    fi
    for task_name in clinc150 banking77
    do
        for lr in 2e-4 1e-4 5e-5
        do
            echo $model_name
            echo $task_name
            echo $lr
            for p_dim in $p_dim_list
            do
                if [ $task_name = clinc150 ]
                then
                    deepspeed --include localhost:0,1 --master_port 29493 main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                    deepspeed --include localhost:0,1 --master_port 29493 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                else
                    deepspeed --include localhost:0,1 --master_port 29493 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                fi
            done
        done
    done
done
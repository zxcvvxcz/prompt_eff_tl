# deepspeed --num_gpus 1 main_ood.py --overwrite_output_dir --split --task_name clinc150 --model_name_or_path gpt2 --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_lora --lora_r 3
# deepspeed --include localhost:2 --master_port 29499 main_ood.py --overwrite_output_dir --split --task_name banking77 --model_name_or_path gpt2-medium --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_adapter --adapter_size 4
# deepspeed --include localhost:2 --master_port 29499 main_ood.py --overwrite_output_dir --split --task_name banking77 --model_name_or_path gpt2-medium --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/banking77/0.25/gpt2 --num_train_epochs 40 --cache_dir ~/data/model_data --lr 5e-6 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 400 --split_ratio 0.25 --apply_adapter --adapter_size 4

master_port=29498
localhost=0,1,2
for model_name in EleutherAI/gpt-neo-2.7B
# for model_name in EleutherAI/gpt-j-6B gpt2 gpt2-medium gpt2-xl gpt2-large EleutherAI/gpt-neo-2.7B
do
    if [ $model_name = gpt2 ]
    then
        r_list="3 17 34"
        a_list="3 17 34"
        p_dim_list="5 31 64"
    elif [ $model_name = gpt2-medium ]
    then 
        r_list="4 18 36"
        a_list="4 18 36"
        p_dim_list="6 34 70"
    elif [ $model_name = gpt2-large ]
    then
        r_list="4 21 42"
        a_list="4 21 42"
        p_dim_list="7 41 83"
    elif [ $model_name = gpt2-xl ]
    then
        r_list="5 25 51"
        a_list="5 25 51"
        p_dim_list="9 50 100"
    elif [ $model_name = EleutherAI/gpt-j-6B ]
    then
        # r_list="13 64 129"
        r_list="13"
        a_list="12 63 128"
        # a_list="63 128"
        p_dim_list="24 125 252"
    elif [ $model_name = EleutherAI/gpt-neo-2.7B ]
    then 
        r_list="8 41 82"
        a_list="8 40 81"
        # p_dim_list="12 14 16 18" # 16 is best
        p_dim_list="76 78 80 82" # 76,79, 82
        # p_dim_list="152 154 156 158 162 164 166 168" # 158 is best
        # p_dim_list="160 79 15"
    fi
    for task_name in banking77
    do
        # for lr in 1e-4 2e-4 5e-5 1e-3
        # for lr in 2e-4 5e-5 1e-3
        for lr in 2e-3 1e-3 5e-4 8e-4 
        do
            for seed in 42
            do
                echo $task_name, $model_name, $lr, $seed
                # for a in $a_list
                # do
                #     if [ $task_name = clinc150 ]
                #     then
                #         deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_adapter --adapter_size $a
                #         deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_adapter --adapter_size $a
                #     else
                #         deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_adapter --adapter_size $a
                #     fi
                # done
                # for r in $r_list
                # do
                #     if [ $task_name = clinc150 ]
                #     then
                #         deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_lora --lora_r $r
                #         deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_lora --lora_r $r

                #     else
                #         deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_lora --lora_r $r
                #     fi
                # done
                for p_dim in $p_dim_list
                do
                    if [ $task_name = clinc150 ]
                    then
                        deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/full/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                        deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                    else
                        deepspeed --include localhost:$localhost --master_port $master_port main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_prefix --num_prefix 10 --mid_dim $p_dim
                    fi
                done
            done
        done
    done
done

# deepspeed --include localhost:2 --master_port 24988 main_ood.py --overwrite_output_dir --task_name banking77 --split --split_ratio 0.25 --model_name_or_path EleutherAI/gpt-neo-2.7B --ds_config ds_configs_samples/zero1_config.json --output_dir outputs/banking77/0.25/EleutherAI/gpt-neo-2.7B --num_train_epochs 40 --cache_dir ~/data/model_data --lr 8e-4 --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_adapter --adapter_size 81
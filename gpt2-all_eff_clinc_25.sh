# lora, prefix, adapter
# for model_name in gpt2 gpt2-medium gpt2-large gpt2-xl
for model_name in gpt2-xl
do
    if [ $model_name = gpt2 ]
    then
        r_list="3 16 34"
        a_list="2 16 33"
    elif [ $model_name = gpt2-medium ]
    then 
        r_list="3 18 36"
        a_list="3 18 36"
    elif [ $model_name = gpt2-large ]
    then
        r_list="4 21 42"
        a_list="4 20 42"
    elif [ $model_name = gpt2-xl ]
    then
        r_list="5 25 51"
        a_list="4 25 51"
    fi
    for task_name in clinc150 banking77
    do
        for lr in 1e-4 2e-4 5e-5
        do
            echo $model_name
            echo $task_name
            echo $lr
            for r in $r_list
            do
                if [ $task_name = clinc150 ]
                then
                    deepspeed --include localhost:2 --master_port 25001 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config_single_gpu.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_lora --lora_r $r
                # else
                #     deepspeed --include localhost:2 --master_port 25001 main_ood.py --overwrite_output_dir --task_name $task_name --split --split_ratio 0.25 --model_name_or_path $model_name --ds_config ds_configs_samples/zero2_config.json --output_dir outputs/$task_name/0.25/$model_name --num_train_epochs 40 --cache_dir ~/data/model_data --lr $lr --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --apply_lora --lora_r $r
                fi
            done
        done
    done
done
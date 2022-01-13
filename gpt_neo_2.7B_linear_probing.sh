output_dir="/home/heyjoonkim/data"
learning_rates="9e-4"
prompt_lengths="5"


prompt_lengths="20"
learning_rates="5e-3"
for prompt_length in $prompt_lengths; do
    for learning_rate in $learning_rates; do
        deepspeed main_ood.py \
            --task_name clinc150 \
            --overwrite_output_dir \
            --model_name_or_path EleutherAI/gpt-neo-2.7B \
            --ds_config ds_configs_samples/zero2_config.json \
            --output_dir $output_dir/linear_probing/ \
            --num_train_epochs 40 \
            --cache_dir model_data \
            --lr $learning_rate \
            --weight_decay 0.1 \
            --lr_scheduler_type cosine \
            --num_warmup_steps 4000 \
            --apply_linear_probing
    done
done
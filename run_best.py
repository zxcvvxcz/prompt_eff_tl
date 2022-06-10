import pandas as pd
import os
import pdb
def main():
    output_paths = [('banking77', '0.25'), ('clinc150', 'full'), ('clinc150', '0.25')]
    for task_name, ratio in output_paths:
        output_csv = os.path.join('outputs', task_name, ratio, 'best_results.csv')
        df = pd.read_csv(output_csv).sort_values(['model', 'eff_method', 'hyperparam_1', 'hyperparam_2', 'maha accuracy'],
                                                 ascending=[True, True, True, True, False])
        df_max = df.groupby(['model', 'eff_method', 'hyperparam_1', 'hyperparam_2']).max(['maha_accuracy'])
        with open(f'run_best_{task_name}_{ratio}.sh', 'w') as f:
            for idx, row in df_max.iterrows():
                model, eff_method, hp1, hp2 = idx
                # if 'gpt-j' in model:
                #     continue
                if 'gpt-j' in model or 'gpt-neo' in model:
                    model = 'EleutherAI/' + model[:-1] + 'B'
                lr = row['lr']
                master_port = 24998
                localhost='1,2'
                for seed in [1, 7, 123, 1234]:
                    run_command = f'deepspeed --include localhost:{localhost} --master_port {master_port} main_ood.py --overwrite_output_dir --task_name {task_name} --model_name_or_path {model} --output_dir outputs_best/{task_name}/full/{model} --ds_config ds_configs_samples/zero2_config.json --num_train_epochs 40 --cache_dir ~/data/model_data --lr {lr} --weight_decay 0.1 --lr_scheduler_type cosine --num_warmup_steps 200 --seed {seed} '
                    if ratio != 'full':
                        run_command += f'--split --split_ratio {ratio} '
                    if eff_method == 'adapter':
                        run_command += f'--apply_adapter --adapter_size {hp1}'
                    elif eff_method == 'lora':
                        run_command += f'--apply_lora --lora_r {hp1}'
                    elif eff_method == 'prefix':
                        run_command += f'--apply_prefix --num_prefix {hp2} --mid_dim {hp1}'
                        
                    f.write(run_command + '\n')
    
if __name__ == '__main__':
    main()

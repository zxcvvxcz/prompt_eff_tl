import csv
import pandas as pd
import os
import pdb
import glob
import sys
import re
import numpy as np

def main():
    datasets = [
                os.path.join('outputs_best', 'banking77', '0.25'),
                os.path.join('outputs_best', 'clinc150', 'full'),
                os.path.join('outputs_best', 'clinc150', '0.25'),
                ]
    for dataset in datasets:
        results_path = os.path.join(dataset, 'best_results.csv')
        with open(results_path, 'w') as f:
            wr = csv.writer(f)
            title = ['model', 'eff_method', 'hyperparam_1', 'hyperparam_2', 'lr', 'seed',
                    'AUROC(cosine)', 'AUROC(energy)', 'AUROC(maha)', 'AUROC(softmax)',
                    'FPR-95(cosine)', 'FPR-95(energy)', 'FPR-95(maha)', 'FPR-95(softmax)',
                    'accuracy', 'maha accuracy']
            wr.writerow(title)
            for filename in glob.iglob(dataset + '**/**/*[0-9].tsv', recursive=True):
                print(filename)
                try:
                    # new_filename = filename[:re.search(r'eval_result', filename, re.I).end()] + filename[re.search(r'eval_result', filename, re.I).end() + 8:]
                    # os.rename(filename, new_filename)
                    df = pd.read_csv(filename, delimiter='\t')
                    best_result = df.iloc[df.iloc[:, -1].idxmax()].tolist()
                    
                    dir_split = filename.split('/')
                    model = dir_split[4] if dir_split[3] == 'EleutherAI' else dir_split[3]
                    filename_no_path = dir_split[-1]
                    
                    filename_split = filename_no_path.split('_')
                    re.search(r'eval_result', filename, re.I).end()
                    seed = filename_split[3]
                    lr_end = re.search(r'[alnf]', filename_split[4], re.I).start() # lr 마지막 숫자 바로 다음 글자
                    lr = filename_split[4][:lr_end]

                    if 'adapter' in filename_no_path:
                        eff_method = 'adapter'
                        hyperparam1 = filename_split[-1][:-4]
                        hyperparam2 = 0
                    elif 'lora' in filename_no_path:
                        eff_method = 'lora'
                        hyperparam1 = filename_split[-4]
                        hyperparam2 = filename_split[-1][:-4]
                    elif 'prefix' in filename_no_path:
                        eff_method = 'prefix'
                        hyperparam1 = filename_split[-1][:-4]
                        hyperparam2 = filename_split[-4]
                    else: # fine-tuning
                        eff_method = 'fine-tune'
                        hyperparam1 = 0
                        hyperparam2 = 0
                        
                    row = [model, eff_method, hyperparam1, hyperparam2, lr, seed] + best_result
                    wr.writerow(row)

                except:
                    # 너무 안 좋아서 log 안 찍힌 경우 일부 있음
                    print(f'Error on {filename}', file=sys.stderr)
                    # shutil.move(filename, filename[:-4] + "_error.tsv")

        # write summary (mean, std) to csv file
        df = pd.read_csv(results_path)
        df2 = pd.read_csv(results_path[:7] + results_path[12:])
        df3 = pd.concat([df, df2], ignore_index=True)
        df4 = df3.groupby(['model','eff_method','hyperparam_1','hyperparam_2','lr']).agg(['mean', np.std, 'count'])
        df5 = df4[df4['accuracy', 'count'] >= 5]
        for col in df5.columns:
            if 'seed' in col or ('count' in col and 'maha accuracy' not in col):
                df5.pop(col)
        # df_save = df5.groupby(['model','eff_method','hyperparam_1','hyperparam_2','lr']).agg(['mean', np.std])
        df_save = df5
        df_save.to_csv(os.path.join(dataset, 'summary.csv'))

if __name__ == '__main__':
    main()

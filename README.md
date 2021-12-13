# Efficient transfer learning

## Installation
* Install requirements
  * pip install -r requirements
* Install DeepSpeed
  * DDP~ZeRO1 -> pip install deepspeed
  * DDP~ZeRO3 -> install from source https://www.deepspeed.ai/tutorials/advanced-install/
* Install custom transformers
  * pip install -e .

## How to run
* e.g. deepspeed main.py --task_name sst2 --model_name_or_path gpt2-xl --ds_config ds_configs/ddp.json --output_dir OUTPATH
## Implementation & Tests

|Method         | DDP   | FP16  | ZeRO1 | ZeRO2 | ZeRO3 |
|---            |---    |---    |---    |---    |---    |
|Fine tuning    |O      |O      |O      |O      |O      |
|LoRA           |O      |O      |O      |O      |X      |
|Prefix tuning  |O      |O      |O      |O      |X      |
|Adapter H      |X      |X      |X      |X      |X      |
|Adapter P      |X      |X      |X      |X      |X      |
|Prompt tuning  |X      |X      |X      |X      |X      |



## Benchmark Results
### GPT2-XL (1.5B)
|Method         |PARAM | MNLI 10%   | SST-2 | RTE   |MNLI   |
|---            |---   |---         |---    |---    |---    |
|Fine tuning    |100%  |            |       |       |       |
|LoRA           |      |            |       |       |       |
|Prefix tuning  |      |            |       |       |       |
|Adapter H      |      |            |       |       |       |
|Adapter P      |      |            |       |       |       |
|Prompt tuning  |      |            |       |       |       |
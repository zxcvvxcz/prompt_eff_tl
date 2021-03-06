""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import json
from functools import partial
import datasets
from datasets import load_dataset, load_metric, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
from time import time


import transformers
from transformers.deepspeed import HfDeepSpeedConfig, deepspeed_config
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
    get_scheduler,
    set_seed,
)
import torch
import torch.nn.functional as F
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.utils import see_memory_usage
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, GPT2Tokenizer, GPT2Model, GPTNeoModel, GPTJModel, T5Tokenizer, T5Model, T5EncoderModel, T5ForConditionalGeneration

from model_wrapper.GPT2Wrapper import GPT2Wrapper
from model_wrapper.T5Wrapper import T5EncWrapper
from model_wrapper.InputProcessor import *
from model_wrapper.OutputProcessor import BaseOutputProcessor
from utils import save_config, set_value_to_shared_json_file, get_value_from_shared_json_file

from ood_utils import load_intent_datasets, preprocess_dataset_for_transformers, collate_fn, check_ood_eval_condition
from ood_eval import prepare_ood, get_maha_embedding
logger = logging.getLogger(__name__)
start = time()
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    'clinc150': ("text", None),
    'snips': ("text", None),
    'banking77': ("text", None),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_train_samples",
        default=None,
        help="Maximum train samples to use at train time, slice from raw train dataset for fast experiment purpose",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--load_init_model",
        action="store_true",
        help="If passed, initialize model from zero checkpoint.",
    )
    parser.add_argument(
        "--save_init_model",
        action="store_true",
        help="If passed, save initial model as zero checkpoint.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join("ckpt"),
        help="If passed, save initial model as zero checkpoint.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=20, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "cosine_custom"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=1000, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        '--overwrite_output_dir', 
        default=False, 
        action="store_true",
        help='Overwrite output directory.'
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        '--ds_config', 
        default='ds_config.json', 
        type=str, 
        help='deepspeed config'
    )
    parser.add_argument(
        '--local_rank', 
        default=0, 
        type=int, 
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--apply_lora', 
        default=False, 
        action="store_true",
        help='apply LoRA params'
    )
    parser.add_argument(
        '--lora_alpha', 
        default=16, 
        type=int, 
        help='LoRA alpha'
    )
    parser.add_argument(
        '--lora_r', 
        default=8, 
        type=int, 
        help='LoRA r'
    )
    parser.add_argument(
        '--apply_prefix', 
        default=False, 
        action="store_true",
        help='apply prefix tuning params'
    )
    parser.add_argument(
        '--num_prefix', 
        default=10, 
        type=int, 
        help='number of prefix to append per layer'
    )
    parser.add_argument(
        '--mid_dim', 
        default=16, 
        type=int, 
        help='reparameterization dim'
    )
    parser.add_argument(
        '--apply_adapter', 
        default=False, 
        action="store_true",
        help='apply adapter tuning params'
    )
    parser.add_argument(
        '--adapter_size', 
        default=2,
        type=int, 
        help='size of adapter'
    )
    parser.add_argument(
        '--adapter_type', 
        default='houlsby',
        type=str,
        help='type of adapter(houlsby, pfeiffer)'
    )

    ## OURS ##
    parser.add_argument(
        '--apply_encoder', 
        default=False, 
        action="store_true",
        help='Apply input dependent encoder.'
    )
    parser.add_argument(
        '--apply_input', 
        default=False, 
        action="store_true",
        help='Apply input for prompt generating.'
    )
    parser.add_argument(
        '--encoder_model_name_or_path', 
        default='gpt2', 
        type=str, 
        help='PLM for encoder.'
    )
    parser.add_argument(
        '--freeze_encoder', 
        default=False, 
        action="store_true",
        help='Freeze PLM for the encoder.'
    )
    parser.add_argument(
        '--apply_prompt', 
        default=False, 
        action="store_true",
        help='apply prompt tuning'
    )
    parser.add_argument(
        '--prompt_length', 
        default=None, 
        type=int, 
        help='Number of prompt tokens.'
    )
    parser.add_argument(
        '--reparameterize', 
        default=False, 
        action="store_true",
        help='Reparameterize prompt.'
    )
    parser.add_argument(
        '--cache_dir', 
        default=None,
        type=str, 
        help='Where do you want to store the pretrained models downloaded from huggingface.co'
    )
    parser.add_argument(
        '--apply_linear', 
        default=False, 
        action="store_true",
        help='apply linear evaluation'
    )
    
    # OOD
    parser.add_argument(
        '--split', 
        action="store_true",
        help='Split setting for intent datasets.'
    )
    parser.add_argument(
        '--split_ratio', 
        default=0.25, 
        type=float, 
        help='Split ratio for intent datasets.'
    )
    parser.add_argument(
        '--lr_ratio', 
        default=0.2, 
        type=float, 
        help='Learning ratio for custom cosine lr scheduler, must be between 0.1~0.2.'
    )


    args = parser.parse_args()
    
    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    # post init get batch and zero option from ds config
    with open(args.ds_config, "r", encoding="utf-8") as ds_f:
        ds_config = json.load(ds_f)
    args.per_device_batch_size = ds_config['train_micro_batch_size_per_gpu']
    if args.gradient_accumulation_steps != ds_config['gradient_accumulation_steps']:
        print('gradient_accumulation_steps mismatch!')
        if args.gradient_accumulation_steps == 1 or ds_config['gradient_accumulation_steps'] != 1:
            args.gradient_accumulation_steps = ds_config['gradient_accumulation_steps']
        else:
            ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if ds_config.get("zero_optimization"):
        args.is_zero3 = ds_config["zero_optimization"]["stage"] == 3
    else:
        args.is_zero3 = False
        
    return args


def main():
    args = parse_args()
    dschf = HfDeepSpeedConfig(args.ds_config)
    deepspeed.init_distributed()
    args.world_size = torch.distributed.get_world_size()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if args.local_rank == 0 else logging.ERROR)
    # logger.setLevel(logging.INFO)
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                logger.info(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')
                exit()
            
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)   
    
    if args.local_rank == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation & SummaryWriter
    if args.local_rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        save_config(args)
        writer = SummaryWriter(args.output_dir)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    intent_tasks = ['clinc150', 'snips', 'banking77']
    dataloader_cols = ['input_ids', 'attention_mask', 'label', 'text', 'indices', 'label_text']
    if args.task_name is not None:
        raw_datasets = DatasetDict()
        if args.task_name in ['clinc150', 'snips', 'banking77']:
            dict_datasets, label_list = load_intent_datasets(args.task_name, split=args.split, ratio=args.split_ratio, check_data=False)
            dict_datasets = preprocess_dataset_for_transformers(dict_datasets)
            for k, v in dict_datasets.items():
                raw_datasets[k] = v
        else:
            # Downloading and loading a dataset from the hub.
            if args.max_train_samples is not None:
                raw_train_dataset = load_dataset("glue", args.task_name, split=f'train[:{args.max_train_samples}]')
            else:
                raw_train_dataset = load_dataset("glue", args.task_name, split=f'train')
            # Since glue test set is not opened, use 1K train as validation and original validation as test
            
            # for small datasets (RTE, ...)
            if len(raw_train_dataset) < 10000:
                raw_eval_dataset = load_dataset("glue", args.task_name, split=f'validation')
                eval_test_split = raw_eval_dataset.train_test_split(test_size=0.5)
                raw_datasets['train'] = raw_train_dataset
                raw_datasets['validation'] = eval_test_split['train']
                raw_datasets['test'] = eval_test_split['test']
            # for larger datasets
            else:
                train_test_split = raw_train_dataset.train_test_split(test_size=1000)
                raw_datasets['train'] = train_test_split['train']
                raw_datasets['validation'] = train_test_split['test']
                
                # for mnli 
                if args.task_name == "mnli":
                    raw_datasets['test'] = load_dataset("glue", args.task_name, split='validation_matched')
                # other tasks
                else:
                    raw_datasets['test'] = load_dataset("glue", args.task_name, split=f'validation')
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    if args.local_rank == 0:
        logger.info('TRAIN / VALIDATION / TEST split.')
        for split, dataset in raw_datasets.items():
            logger.info(f'{split} > {len(dataset)}')
            
    # Labels
    if args.task_name is not None and args.task_name not in intent_tasks:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    # TODO: only inject pad_token_id in case of GPT
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, 
        finetuning_task=args.task_name, pad_token_id=tokenizer.unk_token_id,
        apply_lora=args.apply_lora, lora_alpha=args.lora_alpha, lora_r=args.lora_r,
        apply_prefix=args.apply_prefix, num_prefix=args.num_prefix, mid_dim=args.mid_dim,
        apply_encoder=args.apply_encoder, apply_input=args.apply_input, encoder_model_name_or_path=args.encoder_model_name_or_path,
        freeze_encoder=args.freeze_encoder, prompt_length=args.prompt_length, 
        apply_adapter=args.apply_adapter, adapter_size=args.adapter_size, 
        reparameterize=args.reparameterize,
    )
    # TODO : fix?
    if args.is_zero3:
        zero_init_start_time = time()
        see_memory_usage('Before zero init', True)
        with deepspeed.zero.Init(config_dict_or_path=args.ds_config):
            if 't5' in args.model_name_or_path:
                model = T5EncWrapper(config, args.model_name_or_path, args.cache_dir, (args.task_name in intent_tasks), 
                                    args.load_init_model, args.ckpt_path)
            else:
                model = GPT2Wrapper(config, args.model_name_or_path, args.cache_dir, (args.task_name in intent_tasks), 
                                    args.load_init_model, args.ckpt_path)
           
        print(f"Zero init time: {time() - zero_init_start_time}")
    else:
        if 't5' in args.model_name_or_path:
            model = T5EncWrapper(config, args.model_name_or_path, args.cache_dir, (args.task_name in intent_tasks), 
                                args.load_init_model, args.ckpt_path)
        else:
            model = GPT2Wrapper(config, args.model_name_or_path, args.cache_dir, (args.task_name in intent_tasks), 
                                args.load_init_model, args.ckpt_path)
            
        # model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path, cache_dir=args.cache_dir, get_last_hidden_state=(args.task_name in intent_tasks))
    # pdb.set_trace()
   # Preprocessing the datasets
    see_memory_usage('After init', True)
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None:
        if args.local_rank == 0:
            logger.info('Auto label2id, id2label created')
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result
    
    if args.local_rank != 0:
        torch.distributed.barrier()
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    if args.task_name in intent_tasks:
        test_ind_dataset = processed_datasets['test_ind']
        test_ood_dataset = processed_datasets['test_ood']
    else:
        test_dataset = processed_datasets["test"]
        
    if args.local_rank == 0:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        # data_collator = DataCollatorWithPadding(tokenizer)
        collate_fn_partial = partial(collate_fn, pad_token_id=config.pad_token_id)
        data_collator = collate_fn_partial

    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    maha_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    if args.task_name in intent_tasks:
        test_ind_sampler = DistributedSampler(test_ind_dataset)
        test_ind_dataloader = DataLoader(test_ind_dataset, sampler=test_ind_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)       
        test_ood_sampler = DistributedSampler(test_ood_dataset)
        test_ood_dataloader = DataLoader(test_ood_dataset, sampler=test_ood_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)       
    
    else:
        eval_sampler = DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)
        test_sampler = DistributedSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=data_collator, batch_size=args.per_device_batch_size, shuffle=False)       
    
    # math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # for mahalanobis
    label_id_list = []
    for train_data in train_dataset:
        label = train_data['labels']
        if label not in label_id_list:
            label_id_list.append(label)
    label_id_list.sort()
    
    # Get the metric function
    experiment_id = os.environ["MASTER_PORT"] # since master port should be different for multiple execution, it can serve as experiment id

    if args.task_name in intent_tasks:
        ood_metric_energy = load_metric('OOD', 'energy', num_process=args.world_size, process_id=args.local_rank, experiment_id=experiment_id)
        ood_metric_softmax = load_metric('OOD', 'softmax', num_process=args.world_size, process_id=args.local_rank, experiment_id=experiment_id)
        ood_metric_maha = load_metric('OOD', 'maha', num_process=args.world_size, process_id=args.local_rank, label_id_list=label_id_list, experiment_id=experiment_id)
        ood_metric_cosine = load_metric('OOD', 'cosine', num_process=args.world_size, process_id=args.local_rank, experiment_id=experiment_id)
        ind_metric = load_metric('accuracy', num_process=args.world_size, process_id=args.local_rank, experiment_id=experiment_id)
        ind_metric_maha = load_metric('IND', 'maha_acc', num_process=args.world_size, process_id=args.local_rank, label_id_list=label_id_list, experiment_id=experiment_id)
        metrics = [ind_metric, ind_metric_maha, ood_metric_softmax, ood_metric_energy, ood_metric_cosine, ood_metric_maha]
    elif args.task_name is not None:
        metric = load_metric('glue', args.task_name, num_process=args.world_size, process_id=args.local_rank, experiment_id=experiment_id)
    else:
        metric = load_metric("accuracy", num_process=args.world_size, process_id=args.local_rank, experiment_id=experiment_id)

    # Set params to train
    trainable_param_names = []
    if args.apply_lora:
        trainable_param_names.append('lora')
    if args.apply_prefix:
        trainable_param_names.append('prefix')
    if args.apply_adapter:
        trainable_param_names.append('adapter')
    if args.apply_encoder:
        trainable_param_names.append('encoder')
        
    if len(trainable_param_names) > 0 or args.apply_linear:
        for name, param in model.named_parameters():
            # train main model? (== fine-tuning)
            if name.startswith('transformer'):
                param.requires_grad = False
                for trainable_param_name in trainable_param_names:
                    if trainable_param_name in name:
                        if args.local_rank == 0:
                            logger.info(f'>> TRAIN {name} {param.shape} -> {param.numel()}')
                        param.requires_grad = True
            else:
                # train PLM encoder?
                if "input_processor.encoder." in name:
                    if args.freeze_encoder:
                        param.requires_grad = False
                    else: 
                        param.requires_grad = True
                        if args.local_rank == 0:
                            logger.info(f'>> TRAINED ENCODER {name} {param.shape} -> {param.numel()}')
                else:
                    param.requires_grad = True
                    if args.local_rank == 0:
                        logger.info(f'>> OTHERS {name} {param.shape} -> {param.numel()}')
                
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad==True],
            "weight_decay": 0.0,
        },
    ]
    
    if args.local_rank == 0:
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_trainable_params_except_linear = num_trainable_params - num_labels * model.transformer.wte.embedding_dim
        num_total_params = sum(p.numel() for p in model.parameters())
        transformer_params = sum(p.numel() for n,p in model.named_parameters() if n.startswith('transformer'))
        # logger.info(f'trainable params {num_trainable_params} / total params {num_total_params} = ratio {100 * num_trainable_params/num_total_params} ')
        
        ## Write parameter info ##
        parameter_summary_file = os.path.join(args.output_dir, "parameter_summary.txt")
        with open(parameter_summary_file, "w") as file_writer:
            file_writer.write("Overall Parameter Summary\n")
            file_writer.write(f"Trained     parameters\t{num_trainable_params}\n")
            file_writer.write(f"Transformer parameters\t{transformer_params}\n")
            file_writer.write(f"Total       parameters\t{num_total_params}\n")
            # file_writer.write(f"Trainable   ratio\t\t{100 * num_trainable_params / num_total_params} \n")
            file_writer.write("=" * 50 + '\n')
            file_writer.write("Trained parameters detail\n")

            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    file_writer.write(f"{name} > {param.shape} \n")
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        lr_ratio=args.lr_ratio
    )
    see_memory_usage('Before model engine', True)
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, config=args.ds_config)
    if args.load_init_model:
        load_path, client_sd = model_engine.load_checkpoint(args.ckpt_path)
        print(f'Loaded initial model from {load_path}')
    if args.save_init_model:
        model_engine.save_checkpoint(args.ckpt_path)
        print(f'Saved initial model in {load_path}')
    see_memory_usage('After model engine', True)
    # pdb.set_trace()
    # del model
    # see_memory_usage('After delete model', True)
    # Train!
    if args.local_rank == 0:
        total_batch_size = args.per_device_batch_size * args.world_size * args.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  World Size = {args.world_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Random Seed = {args.seed}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Number of trainable params = {num_trainable_params}")
        logger.info(f"  Number of trainable params by eff method = {num_trainable_params_except_linear}")
        logger.info(f"  Number of total params = {num_total_params}")
        logger.info(f"  Number of PLM params = {transformer_params}")
        logger.info(f"  % of trainable params per whole model (old) = {(100 * num_trainable_params/num_total_params):.3f}")
        logger.info(f"  % of trainable params per PLM only (new) = {(100 * num_trainable_params_except_linear/transformer_params):.3f}")
        if args.apply_lora:
            eff_method = 'lora'
            hyperparams = [args.lora_r, args.lora_alpha]
        elif args.apply_adapter:
            eff_method = 'adapter'
            hyperparams = [args.adapter_size, 0]
        elif args.apply_prefix:
            eff_method = 'prefix'
            hyperparams = [args.mid_dim, args.num_prefix]
        else:
            eff_method = 'fine_tune'
            hyperparams = [0, 0]
        
        row = [args.model_name_or_path, eff_method, 
            100 * num_trainable_params_except_linear/transformer_params, 
            hyperparams[0], hyperparams[1]
            ]
        if not os.path.exists("param_ratio.csv"):
            with open("param_ratio.csv", 'w') as f:
                wr = csv.writer(f)
                # hyperparam1: lora_r, adapter_size, mid_dim
                # hyperparam2: lora_alpha, None, num_prefix
                title = ['model_name', 'eff_method', 'train_ratio', 'hyperparam_1', 'hyperparam_2']
                
                wr.writerow(title)
                wr.writerow(row)
        else:
            with open("param_ratio.csv", 'a') as f:
                wr = csv.writer(f)
                wr.writerow(row)





if __name__ == "__main__":
    main()
import argparse
import random

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from bitsandbytes.optim import AdamW
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from config.default_arguments import DataTrainingArguments, PeftArguments
from config.qwen_arguments import (
    Qwen32BwithUnslothDataTrainingArguments,
    Qwen32BWithUnslothModelArguments,
)
from src.data.data_loader import load_datasets, load_datasets_for_testset
from src.data.dataset import BaseDataset
from src.data.templates import get_default_chat_template
from src.evaluation.metrics import compute_metrics, preprocess_logits_for_metrics
from src.models.qwen import QwenBaseModelWithUnsloth
from src.training.trainer import Trainer
from src.utils.util import set_seed

pd.set_option("display.max_columns", None)

set_seed(42)  # magic number :)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Configuration file path")

    parser_args = parser.parse_args()
    # Load the train dataset

    with open(parser_args.config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = Qwen32BWithUnslothModelArguments(**config["model"])
    training_args = Qwen32BwithUnslothDataTrainingArguments(**config["training"])
    peft_args = PeftArguments(**config["peft"])

    qwen_model = QwenBaseModelWithUnsloth(model_name=model_args.name, max_seq_length=model_args.max_seq_length, dtype=getattr(torch, model_args.dtype), load_in_4bit=model_args.load_in_4bit)
    model, tokenizer = qwen_model.get_model_and_tokenizer()

    tokenizer.chat_template = get_default_chat_template()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    train_dataset, val_dataset = load_datasets_for_testset(config["data"]["train"]["file_path"], config["data"]["eval"]["file_path"], tokenizer, max_seq_length=training_args.max_seq_length)

    response_template = "<start_of_turn>model"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type="cosine",
        max_seq_length=training_args.max_seq_length,
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.batch_size,
        per_device_eval_batch_size=training_args.batch_size,
        num_train_epochs=training_args.num_epochs,
        max_steps=training_args.max_steps,
        save_steps=training_args.save_steps,
        learning_rate=float(training_args.learning_rate),
        optim=training_args.optim,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_strategy=training_args.save_strategy,
        eval_strategy=training_args.eval_strategy,
        save_total_limit=training_args.save_total_limit,
        save_only_model=True,
        report_to="none",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_args.r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        target_modules=peft_args.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        use_rslora=True,
        loftq_config=None,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda evaluation_result: compute_metrics(evaluation_result, tokenizer),
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, tokenizer),
        sft_config=sft_config,
        peft_config=None,
    )

    # Train the model
    trainer.train()

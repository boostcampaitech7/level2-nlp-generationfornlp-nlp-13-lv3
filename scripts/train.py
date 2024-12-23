import argparse
import json
import random
from ast import literal_eval

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from config.default_arguments import (
    DataTrainingArguments,
    ModelArguments,
    PeftArguments,
)
from src.data.data_loader import load_datasets, load_datasets_V2
from src.data.dataset import BaseDataset
from src.evaluation.metrics import compute_metrics, preprocess_logits_for_metrics
from src.models.gemma import GemmaBaseModel
from src.training.trainer import Trainer
from src.utils.util import set_seed

pd.set_option("display.max_columns", None)

set_seed(42)  # magic number :)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=True, help="Configuration file path")

    parser_args = parser.parse_args()
    # Load the train dataset

    with open(parser_args.config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = ModelArguments(**config["model"])
    training_args = DataTrainingArguments(**config["training"])
    peft_args = PeftArguments(**config["peft"])

    gemma_model = GemmaBaseModel(model_args.name)
    model, tokenizer = gemma_model.get_model_and_tokenizer()

    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.special_tokens_map)
    tokenizer.padding_side = "right"

    train_dataset, val_dataset = load_datasets_V2(config["data"]["train"]["file_path"], tokenizer, max_seq_length=training_args.max_seq_length)

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
        learning_rate=float(training_args.learning_rate),
        weight_decay=training_args.weight_decay,
        logging_steps=training_args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=training_args.save_total_limit,
        save_only_model=True,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=peft_args.r,
        lora_alpha=peft_args.lora_alpha,
        target_modules=peft_args.target_modules,
        lora_dropout=peft_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
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
        peft_config=peft_config,
    )

    # Train the model
    trainer.train()

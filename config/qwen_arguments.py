from dataclasses import dataclass, field
from typing import Optional, List

from config.default_arguments import ModelArguments, DataTrainingArguments


@dataclass
class Qwen32BWithUnsloth_ModelArguments(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    name: str = field(
        default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded."},
    )
    dtype: str = field(
        default="float16",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    load_in_4bit: bool = field(default=True, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})


@dataclass
class Qwen32BwithUnsloth_DataTrainingArguments(DataTrainingArguments):

    max_steps: int = field(default=30, metadata={"help": "train을 진행할 최대 step 수 입니다."})
    save_steps: int = field(default=3, metadata={"help": "체크포인트를 저장할 step 수입니다. (save_steps 진행 별 체크포인트 저장 - ex. 3 step 진행 후 저장, 6 step 진행 후 저장)"})
    save_strategy: str = field(default="steps", metadata={"help": "checkpoint save 정책을 지정합니다. (e.g. steps, epoch)"})
    eval_strategy: str = field(default="epoch", metadata={"help": "eval 정책을 지정합니다. (e.g. steps, epoch)"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "warmup ratio를 지정합니다."})

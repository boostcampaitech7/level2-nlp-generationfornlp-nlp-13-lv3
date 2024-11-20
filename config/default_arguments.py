from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    name: str = field(
        default="beomi/gemma-ko-2b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    batch_size: int = field(
        default=4,
        metadata={"help": "train batch size"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "train learning rate"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer " "than this will be truncated, sequences shorter will be padded."},
    )
    num_epochs: int = field(
        default=1,
        metadata={"help": "train epochs"},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "train weight decay"},
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "logging per steps"},
    )
    save_total_limit: int = field(default=2, metadata={"help": "save limit in training"})
    output_dir: str = field(
        default="experiments/default",
        metadata={"help": "fine tuned model saved directory path"},
    )


@dataclass
class PeftArguments:
    """
    Arguments pertaining for LoRA
    """

    r: int = field(default=8, metadata={"help": "LoRA에서 사용하는 저차원 공간의 랭크(rank)를 지정합니다"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA의 스케일링 계수를 설정합니다"})
    target_modules: list[str] = field(default_factory=lambda: ["query", "value"], metadata={"help": "LoRA에서 사용하는 저차원 공간의 랭크(rank)를 지정합니다"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA의 드롭아웃 확률을 설정합니다"})

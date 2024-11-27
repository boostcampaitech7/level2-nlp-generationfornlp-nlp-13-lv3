from ast import literal_eval

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from src.data.dataset import (
    BaseDataset,
    prepare_data_for_training,
    process_dataset_test,
)
from src.data.preprocessing import prepare_records


def load_datasets(file_path, tokenizer, train_split=0.9):
    """docstring"""
    dataset = pd.read_csv(file_path)

    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }
        records.append(record)

    df = pd.DataFrame(records)

    df["input_text"] = df.apply(lambda x: f"{x['paragraph']} Question: {x['question']} Choices: {', '.join(x['choices'])}", axis=1)

    input_texts = df["input_text"].tolist()
    labels = [int(choice) - 1 for choice in df["answer"]]

    dataset_size = int(train_split * len(input_texts))
    sample_texts, val_texts = input_texts[:dataset_size], input_texts[dataset_size:]
    sample_labels, val_labels = labels[:dataset_size], labels[dataset_size:]

    sample_dataset = BaseDataset(sample_texts, sample_labels, tokenizer)
    val_dataset = BaseDataset(val_texts, val_labels, tokenizer)

    return sample_dataset, val_dataset


def load_datasets_V2(file_path, tokenizer, train_split=0.9, max_seq_length=1024, mode="train", is_augmented=False):
    # 기존 Prompt 정의
    PROMPT_NO_QUESTION_PLUS = """지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"""

    PROMPT_QUESTION_PLUS = """지문:\n{paragraph}\n\n질문:\n{question}\n\n<보기>:\n{question_plus}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"""
    # 데이터 로드 및 준비
    dataset = pd.read_csv(file_path)  # 데이터 경로에 맞게 변경
    # Flatten the JSON dataset
    records = prepare_records(dataset, is_augmented)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    if mode == "train":
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = prepare_data_for_training(dataset, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, tokenizer)

        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length)
        # 데이터셋 분리
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=1.0 - train_split, seed=42)

        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["test"]
    else:
        test_dataset = process_dataset_test(df, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS)
        train_dataset = test_dataset
        eval_dataset = None

    return train_dataset, eval_dataset


def load_datasets_for_testset(train_file_path, eval_file_path, tokenizer, max_seq_length=1024, mode="train"):
    # 기존 Prompt 정의
    PROMPT_NO_QUESTION_PLUS = """지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"""
    PROMPT_QUESTION_PLUS = """지문:\n{paragraph}\n\n질문:\n{question}\n\n<보기>:\n{question_plus}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"""
    # 데이터 로드 및 준비

    train_dataset_df = pd.read_csv(train_file_path)  # 데이터 경로에 맞게 변경
    eval_dataset_df = pd.read_csv(eval_file_path)

    if mode == "train":
        train_dataset = Dataset.from_pandas(train_dataset_df)
        eval_dataset = Dataset.from_pandas(eval_dataset_df)
        train_tokenized_dataset = prepare_data_for_training(train_dataset, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, tokenizer)
        train_tokenized_dataset = train_tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length)

        eval_tokenized_dataset = prepare_data_for_training(eval_dataset, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, tokenizer)
        eval_tokenized_dataset = eval_tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length)

        train_dataset = train_tokenized_dataset
        eval_dataset = eval_tokenized_dataset
    else:
        records = prepare_records(train_dataset_df, is_augmented=True)

        # Convert to DataFrame
        df = pd.DataFrame(records)
        test_dataset = process_dataset_test(df, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS)
        train_dataset = test_dataset
        eval_dataset = None

    return train_dataset, eval_dataset

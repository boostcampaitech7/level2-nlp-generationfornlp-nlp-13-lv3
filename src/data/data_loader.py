import torch
from torch.utils.data import DataLoader
import pandas as pd
from ast import literal_eval
from src.data.dataset import BaseDataset, prepare_data_for_training
from datasets import Dataset


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


def load_datasets_V2(file_path, tokenizer, train_split=0.9):
    # 기존 Prompt 정의
    PROMPT_NO_QUESTION_PLUS = """지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"""

    PROMPT_QUESTION_PLUS = """지문:\n{paragraph}\n\n질문:\n{question}\n\n<보기>:\n{question_plus}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:"""
    # 데이터 로드 및 준비
    dataset = pd.read_csv(file_path)  # 데이터 경로에 맞게 변경
    # Flatten the JSON dataset
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
        # Include 'question_plus' if it exists
        if "question_plus" in problems:
            record["question_plus"] = problems["question_plus"]
        records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = prepare_data_for_training(dataset, PROMPT_NO_QUESTION_PLUS, PROMPT_QUESTION_PLUS, tokenizer)

    # 데이터셋 분리
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=1.0 - train_split, seed=42)

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    return train_dataset, eval_dataset

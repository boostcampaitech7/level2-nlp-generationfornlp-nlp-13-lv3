from ast import literal_eval
from typing import Dict, List, Optional
from xmlrpc.client import Boolean

import pandas as pd


def prepare_dataset_records_with_default(dataset: pd.DataFrame):
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
    return records


def prepare_dataset_records_with_augmented(dataset: pd.DataFrame):
    records = []
    for _, row in dataset.iterrows():
        choices = literal_eval(row["choices"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": row["question"],
            "choices": choices,
            "answer": row["answer"],
            "question_plus": row["question_plus"],
        }
        # Include 'question_plus' if it exists

        records.append(record)

    return records


def prepare_records(dataset: pd.DataFrame, is_augmented: Optional[bool] = False) -> List[Dict]:
    records = []
    if is_augmented:
        records = prepare_dataset_records_with_augmented(dataset)
    else:
        records = prepare_dataset_records_with_default(dataset)

    return records

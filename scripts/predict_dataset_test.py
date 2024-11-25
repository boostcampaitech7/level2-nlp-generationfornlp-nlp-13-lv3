import argparse
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from config.qwen_arguments import (
    Qwen32BwithUnsloth_DataTrainingArguments,
    Qwen32BWithUnsloth_ModelArguments,
)
from src.data.data_loader import load_datasets_V2
from src.data.templates import get_chat_template
from src.models.qwen import QwenBaseModelWithUnsloth
from src.utils.util import get_latest_checkpoint, set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=True, help="Configuration file path")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint folder path")

    parser_args = parser.parse_args()
    # Load the train dataset

    with open(parser_args.config_path, "r") as file:
        config = yaml.safe_load(file)

    model_args = Qwen32BWithUnsloth_ModelArguments(**config["model"])
    training_args = Qwen32BwithUnsloth_DataTrainingArguments(**config["training"])
    ### 체크포인트 인자로 넘겨주면 그것을 최우선으로 사용하고 없으면 모델 + checkpoint 최대 step 폴더를 체크포인트 path로 인식
    if parser_args.checkpoint_path:
        checkpoint_path = parser_args.checkpoint_path
        print("command-line arguments check (checkpoint_path) : ", checkpoint_path)
    else:
        checkpoint_path = get_latest_checkpoint(training_args.output_dir)

        if checkpoint_path:
            print(f"Latest checkpoint: {checkpoint_path}")
        else:
            print("No checkpoints found.")
            sys.exit(0)

    qwen_model = QwenBaseModelWithUnsloth(model_name=checkpoint_path, max_seq_length=model_args.max_seq_length, dtype=getattr(torch, model_args.dtype), load_in_4bit=model_args.load_in_4bit)
    model, tokenizer = qwen_model.get_model_and_tokenizer(inference_mode=True)
    model = model.to(device)  # 모델을 GPU로 이동

    test_dataset, val_dataset = load_datasets_V2(file_path=config["data"]["test"]["file_path"], tokenizer=tokenizer, max_seq_length=training_args.max_seq_length, mode="eval")

    tokenizer.chat_template = get_chat_template()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.special_tokens_map)
    tokenizer.padding_side = "right"

    infer_results = []

    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            # print(data)
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

            probs = torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    pd.DataFrame(infer_results).to_csv(training_args.csv_output_path, index=False)

import argparse
import json
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

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

def merge_json_files(input_dir, output_file):
    """Merge JSON files from WikiExtractor output into a single JSON file."""
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in glob.glob(f"{input_dir}/AA/*.json"):
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

def initialize_retriever(documents):
    """Initialize the Dense Retriever using Ko-Sentence-Transformers."""
    print("Initializing retriever...")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    return model, doc_embeddings

def retrieve_documents(query, retriever_model, doc_embeddings, documents, top_k=3):
    """Retrieve top-k documents relevant to the query."""
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    top_indices = scores.topk(k=top_k).indices
    return [documents[i] for i in top_indices]


if __name__ == "__main__":
    # 기존 초기화, 모델 로드, 데이터 로드 및 retriever 초기화 생략

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_dataset):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            # Combine messages into a query
            query = " ".join([msg["content"] for msg in messages])

            # Retrieve relevant documents
            retrieved_docs = retrieve_documents(query, retriever_model, doc_embeddings, documents, top_k=3)
            retrieved_context = "\n\n".join(retrieved_docs)

            # 프롬프트 강화: 검색된 문서를 문제와 연관지어 설명
            context_prompt = (
                "아래는 문제와 관련된 참고 문서입니다. "
                "이 문서를 참고하여 문제를 해결하고, 정답을 도출하세요.\n\n"
                f"{retrieved_context}\n\n"
                "문제와 관련된 답을 명확히 선택하세요.\n"
            )

            # Augment messages with enhanced context prompt
            augmented_messages = messages + [
                {"role": "system", "content": context_prompt}
            ]

            # Perform inference
            outputs = model(
                tokenizer.apply_chat_template(
                    augmented_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
            )

            # Extract logits and predict answer
            logits = outputs.logits[:, -1].flatten().cpu()
            target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]
            probs = torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32)).detach().cpu().numpy()
            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]

            infer_results.append({"id": _id, "answer": predict_value})

    # Save results to CSV
    pd.DataFrame(infer_results).to_csv(training_args.csv_output_path, index=False)
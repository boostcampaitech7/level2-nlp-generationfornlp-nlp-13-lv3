import torch
import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
import json
import pandas as pd
import random
import numpy as np
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig
#from unsloth import FastLanguageModel


def set_random_seed(seed):
    """난수 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_and_preprocess_data(csv_file):
    """데이터 로드 및 전처리"""
    dataset = pd.read_csv(csv_file)

    # Flatten the JSON dataset
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            'question_plus': problems.get('question_plus', None)
        }
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    
    return df

def process_dataset(df, tokenizer):
    """모델 추론"""
    PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

    PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""
    
    processed_dataset = []
    for _, row in df.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])
        
        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        processed_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )
    return Dataset.from_pandas(pd.DataFrame(processed_dataset))


def inference(model, tokenizer, test_df):
    
    infer_results = []
    pred_choices_map = {i: str(i + 1) for i in range(5)}

    model.eval()
    with torch.inference_mode():
        for data in tqdm(test_df):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()
            vocab = tokenizer.get_vocab()
            target_logit_list = [logits[vocab.get(str(i + 1))] for i in range(len_choices)]

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(target_logit_list, dtype=torch.float32)
                )
                .detach()
                .cpu()
                .numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs,axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})
    
    return infer_results



def save_results(infer_results, output_file):
    """결과 저장"""
    pd.DataFrame(infer_results).to_csv(output_file, index=False)

def main():
    set_random_seed(42)
    checkpoint_path = "outputs_qwen/checkpoint-150"
    """

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True,
    )
    

    tokenizer.chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% endif %}"
        "{% if system_message is defined %}"
        "{{ system_message }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ content + '<end_of_turn>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    )
    #model = FastLanguageModel.for_inference(model)



    """
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    
    test_df = load_and_preprocess_data('test.csv')
    processed_dataset = process_dataset(test_df, tokenizer)

    infer_results = inference(model, tokenizer, processed_dataset)
    save_results(infer_results, "qwq32b.csv")


if __name__ == '__main__':
    main()
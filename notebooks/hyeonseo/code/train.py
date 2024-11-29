import os
import torch
import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoTokenizer
from datasets import Dataset
import json
import pandas as pd
import random
import numpy as np
import evaluate
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

pd.set_option('display.max_columns', None)

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
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    return df

def load_model_and_tokenizer(max_seq_length=4096, dtype=torch.float16, load_in_4bit=True):
    """모델과 토크나이저 로드"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/QwQ-32B-Preview-bnb-4bit",  # 모델 이름을 설정합니다.unsloth/Qwen2.5-32B-Instruct-bnb-4bit
        max_seq_length=max_seq_length,    # 최대 시퀀스 길이를 설정합니다.
        dtype=dtype,                      # 데이터 타입을 설정합니다.
        load_in_4bit=load_in_4bit,        # 4bit 양자화 로드 여부를 설정합니다.
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
        
    # LoRA 적용된 모델 생성
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        use_rslora=True,
        loftq_config=None,
    )
    return model, tokenizer

def process_dataset(df, tokenizer):
    """데이터셋 처리 및 메시지 형식으로 변환"""
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
                    {"role": "assistant", "content": f"{row['answer']}"}
                ],
                "label": row["answer"],
            }
        )
    
    return Dataset.from_pandas(pd.DataFrame(processed_dataset))

def tokenize_dataset(dataset, tokenizer):
    """데이터 토큰화"""
    def formatting_prompts_func(example):
        output_texts = []
        for message in example["messages"]:
            output_texts.append(
                tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                )
            )
        return output_texts
    
    def tokenize_function(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    return tokenized_dataset

def split_dataset(tokenized_dataset):
    """데이터셋 분할"""
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 2048)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']
    return train_dataset, eval_dataset

def get_data_collator(tokenizer):
    """Data Collator 생성"""
    response_template = "<start_of_turn>model"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    return data_collator

def get_trainer(model, tokenizer, train_dataset, eval_dataset, data_collator):
    """SFTTrainer 생성"""
    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        vocab = tokenizer.get_vocab()
        logit_idx = [vocab.get(str(i), tokenizer.unk_token_id) for i in range(1, 6)]
        logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
        return logits

    acc_metric = evaluate.load("accuracy")
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    def compute_metrics(evaluation_result):
        logits, labels = evaluation_result

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))


        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        acc = acc_metric.compute(predictions=predictions, references=labels)
        return acc

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type="cosine",
        max_seq_length=2048,#2048
        output_dir="outputs_qwen",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        #num_train_epochs=2,
        max_steps = 200,
        #warmup_ratio = 0.05,
        warmup_steps = 20,
        learning_rate=5e-5,
        #optim="adamw_8bit",#기본으로 변경
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        #eval_strategy="steps",
        #eval_steps=200,
        save_total_limit=6,
        save_only_model=True,
        report_to="none",
        gradient_accumulation_steps=4,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=sft_config,
    )
    return trainer

def main():
    set_random_seed(42)
    #df = load_and_preprocess_data('original_train.csv')
    df = pd.read_csv('clean_train.csv')
    model, tokenizer = load_model_and_tokenizer()
    processed_dataset = process_dataset(df, tokenizer)
    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer)
    train_dataset, eval_dataset = split_dataset(tokenized_dataset)
    data_collator = get_data_collator(tokenizer)
    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset, data_collator)
    """
    resume_training = False
    if os.path.exists(trainer.args.output_dir) and os.listdir(trainer.args.output_dir):
        resume_training = True
        print("Resuming training from checkpoint.")
    else:
        print("Starting training from scratch.")
    
    # Pass resume_from_checkpoint to trainer.train()
    trainer.train(resume_from_checkpoint=resume_training)
    """
    trainer.train()



if __name__ == '__main__':
    main()

from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import is_bfloat16_supported, FastLanguageModel
import torch


class QwenBaseModelWithUnsloth:
    def __init__(self, model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit", max_seq_length=4096, dtype=torch.float16, load_in_4bit=True):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,  # 모델 이름을 설정합니다.
            max_seq_length=max_seq_length,  # 최대 시퀀스 길이를 설정합니다.
            dtype=dtype,  # 데이터 타입을 설정합니다.
            load_in_4bit=load_in_4bit,  # 4bit 양자화 로드 여부를 설정합니다.
        )

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

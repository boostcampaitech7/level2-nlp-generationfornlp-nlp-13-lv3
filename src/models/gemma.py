import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaBaseModel:
    def __init__(self, model_name="beomi/gemma-ko-2b"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  ## 초기화 변수에 수정
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer


class GemmaBaseModelWithUnsloth:
    def __init__(self, model_name="beomi/gemma-ko-2b"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  ## 초기화 변수에 수정
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

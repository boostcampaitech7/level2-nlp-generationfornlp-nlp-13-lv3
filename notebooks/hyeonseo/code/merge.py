from transformers import AutoModelForCausalLM
from peft import PeftModel
from unsloth import FastLanguageModel
import torch

# 1. 4비트 양자화를 사용하여 기본 모델 로드
"""
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    load_in_4bit=True,  # 4비트 양자화 활성화
    device_map="auto"   # 효율적인 디바이스 매핑
)
"""
base_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",  # 모델 이름을 설정합니다.unsloth/Qwen2.5-32B-Instruct-bnb-4bit
        max_seq_length=4096,    # 최대 시퀀스 길이를 설정합니다.
        dtype=None,                      # 데이터 타입을 설정합니다.
        load_in_4bit=True,        # 4bit 양자화 로드 여부를 설정합니다.
    )
        
    # LoRA 적용된 모델 생성


model = FastLanguageModel.get_peft_model(
        base_model,
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



# 3. 두 번째 어댑터 로드
"""
model.load_adapter(
    "output_old/checkpoint-457",
    adapter_name="adapter1",
    is_trainable=False,       # 필요 없는 그래디언트 추적 비활성화
    torch_dtype=torch.float16 # 데이터 타입 최적화
)
model.load_adapter(
    "output_old/checkpoint-225",
    adapter_name="adapter2",
    is_trainable=False,       # 필요 없는 그래디언트 추적 비활성화
    torch_dtype=torch.float16 # 데이터 타입 최적화
)


"""
model.load_adapter(
    "outputs_qwen/checkpoint-600",
    adapter_name="adapter3",
    is_trainable=False,       # 필요 없는 그래디언트 추적 비활성화
    torch_dtype=torch.float16 # 데이터 타입 최적화
)
model.load_adapter(
    "outputs_qwen/checkpoint-800",
    adapter_name="adapter4",
    is_trainable=False,       # 필요 없는 그래디언트 추적 비활성화
    torch_dtype=torch.float16 # 데이터 타입 최적화
)


# 4. 어댑터 병합
model.add_weighted_adapter(
    adapters=[ "adapter3", "adapter4"],
    weights=[0.6, 0.4],
    adapter_name="merged_adapter",
    combination_type="linear",
)

# 5. 병합된 어댑터 설정
model.set_adapter("merged_adapter")

# 6. 병합된 어댑터 저장
save_path = "output_qwen"
model.save_pretrained(save_path)

print(f"병합된 어댑터가 저장되었습니다: {save_path}")

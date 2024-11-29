import openai
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
import json
import random
import numpy as np
from ast import literal_eval

def set_seed(seed: int = 42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)

class SampleEvaluator:
    def __init__(self, api_key: str):
        # OpenAI API 설정
        openai.api_key = api_key
        
        # 시스템 프롬프트
        self.system_prompt = """당신은 수능 문제 품질 평가 전문가입니다. 
다음 기준으로 문제를 평가하세요:

1. 지문 품질
- 적절성: 문제 해결에 필요한 정보를 충분히 제공하며, 불필요한 정보를 포함하지 않았는가?
- 난이도: 수능 문제로서 적절한 난이도를 유지하며, 지나치게 쉽거나 어려운 수준은 아닌가?

2. 문제 구성
- 질문의 명확성: 무엇을 물어보는지 분명한가?
- 정답에 대한 논리적 근거가 지문 및 질문에서 충분히 도출될 수 있는가?
- 선택지 구성: 선택지의 표현이 명확하며, 학생을 혼란스럽게 만드는 불필요한 요소가 없는가?
- 정답의 타당성: 제시된 정답이 실제로 올바른가?

3. 해설 품질
- 해설이 질문 및 정답과 논리적으로 일치하는가?
- 해설이 학생들이 오답을 해결할 수 있도록 충분한 도움을 주는가?"""

        # 평가 결과를 위한 함수 스키마
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_sample",
                    "description": "수능 문제 샘플의 품질을 평가합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scores": {
                                "type": "object",
                                "properties": {
                                    "text_quality": {
                                        "type": "number",
                                        "minimum": 1,
                                        "maximum": 5,
                                        "description": "지문 품질 점수"
                                    },
                                    "question_quality": {
                                        "type": "number",
                                        "minimum": 1,
                                        "maximum": 5,
                                        "description": "문제 구성 점수"
                                    },
                                    "explanation_quality": {
                                        "type": "number",
                                        "minimum": 1,
                                        "maximum": 5,
                                        "description": "해설 품질 점수"
                                    }
                                },
                                "required": ["text_quality", "question_quality", "explanation_quality"]
                            },
                            "evaluation": {
                                "type": "object",
                                "properties": {
                                    "strengths": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "문제의 장점들"
                                    },
                                    "weaknesses": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "문제의 약점들"
                                    }
                                },
                                "required": ["strengths", "weaknesses"]
                            },
                            "recommendation": {
                                "type": "string",
                                "enum": ["keep", "remove"],
                                "description": "문제 유지 또는 제거 추천"
                            },
                            "reason": {
                                "type": "string",
                                "description": "최종 판단 이유"
                            }
                        },
                        "required": ["scores", "evaluation", "recommendation", "reason"]
                    }
                },
                "strict": True
            }
        ]

    def evaluate_sample(self, row: Dict) -> Dict:
        """개별 샘플 평가"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""다음 수능 문제를 평가해주세요:

지문: {row['paragraph']}
질문: {row['problems']['question']}
선택지: {row['problems']['choices']}
정답: {row['problems']['answer']}
해설: {row.get('question_plus', 'No explanation provided')}"""}
                ],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "evaluate_sample"}},
                max_tokens=2000,
                temperature=0.8
            )

            evaluation = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            evaluation['total_score'] = sum(evaluation['scores'].values()) / 3
            return evaluation

        except Exception as e:
            print(f"Error evaluating sample: {e}")
            print(f"Sample content: {row}")
            return None

    def evaluate_dataset(self, df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
        """전체 데이터셋 평가"""
        evaluated_samples = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating samples"):
            print(f"\nEvaluating sample {idx + 1}/{len(df)}")
            # row를 dictionary로 변환
            row_dict = row.to_dict()
            evaluation = self.evaluate_sample(row_dict)
            
            if evaluation:
                row_dict['evaluation'] = evaluation
                row_dict['total_score'] = evaluation['total_score']
                evaluated_samples.append(row_dict)
        
        if evaluated_samples:
            evaluated_df = pd.DataFrame(evaluated_samples)
            
            # 결과 요약
            print("\n=== Evaluation Summary ===")
            print(f"Total samples evaluated: {len(evaluated_df)}")
            print("\nScore Distribution:")
            print(evaluated_df['total_score'].describe())
            
            return evaluated_df
        else:
            print("No samples were successfully evaluated.")
            return pd.DataFrame()  # 빈 DataFrame 반환

def process_full_dataset(api_key: str, input_file: str, output_file: str):
    """전체 데이터셋 처리"""
    try:
        print("Loading dataset...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} samples")
        
        # problems 컬럼 파싱
        df['problems'] = df['problems'].apply(literal_eval)
        
        # 평가기 초기화 및 평가 실행
        evaluator = SampleEvaluator(api_key)
        evaluated_df = evaluator.evaluate_dataset(df)
        
        # 결과 저장
        evaluated_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY_HERE"
    INPUT_FILE = "train.csv"
    OUTPUT_FILE = "evaluated_train.csv"
    
    process_full_dataset(
        api_key=API_KEY,
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE
    )

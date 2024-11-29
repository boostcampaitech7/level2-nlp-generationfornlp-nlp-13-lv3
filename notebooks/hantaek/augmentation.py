import openai
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

class QuestionAugmenter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.total_tokens = 0
        
        # Tools 정의
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_question",
                    "description": "수능 형식의 새로운 문제를 생성합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paragraph": {
                                "type": "string",
                                "description": "문제의 지문"
                            },
                            "problems": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "문제 질문"
                                    },
                                    "choices": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "minItems": 4,
                                        "maxItems": 5,
                                        "description": "4-5개의 선택지"
                                    },
                                    "answer": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 5,
                                        "description": "정답 번호 (1-5)"
                                    }
                                },
                                "required": ["question", "choices", "answer"],
                                "additionalProperties": False
                            },
                            "question_plus": {
                                "type": "string",
                                "description": "문제 해결을 위한 사고 과정"
                            }
                        },
                        "required": ["paragraph", "problems", "question_plus"],
                        "additionalProperties": False
                    }
                },
                "strict": True
            }
        ]

        self.system_prompt = """당신은 대한민국 수능 출제 위원입니다. 
다음의 출제 경향을 반영하여 문제를 생성해주세요:

1. 교육과정 근거:
- 고등학교 교육과정 내용에 충실
- 핵심 개념과 원리 중심
- 교과서 수준의 난이도 유지

2. 사고력 평가:
- 단순 암기가 아닌 자료 분석력
- 비판적 사고력과 추론 능력
- 문제 해결을 위한 종합적 사고

3. 주제 영역 균형:
- 정치: 민주주의, 제도와 참여
- 경제: 시장 경제, 경제 정책
- 사회문화: 사회 변동, 문화와 다양성
- 윤리: 전통 윤리, 현대 생활 윤리
- 한국사: 정치사, 경제사, 사회문화사"""

    def generate_augmented_question(self, original_data: Dict) -> Dict:
        """수능 출제 경향을 반영한 문제 생성"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""다음 문제를 바탕으로 새로운 문제를 생성해주세요.
                    
원본 문제:
지문: {original_data['paragraph']}
질문: {original_data['problems']['question']}
선택지: {original_data['problems']['choices']}
정답: {original_data['problems']['answer']}

핵심 개념을 활용하여 새로운 문제를 만들어주세요."""}
                ],
                tools=self.tools, 
                tool_choice={"type": "function", "function": {"name": "generate_question"}},
                temperature=0.7,
                max_tokens=1500
            )
            
            if 'usage' in response:
                self.total_tokens += response['usage']['total_tokens']

            
            if hasattr(response.choices[0].message, 'tool_calls'):
                tool_response = eval(response.choices[0].message.tool_calls[0].function.arguments)
                return tool_response
            else:
                print("No tool calls in response")
                return None
            
        except Exception as e:
            print(f"\nError in generate_augmented_question: {str(e)}")
            return None

    def augment_dataset(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """데이터셋 증강"""
        if sample_size:
            df = df.sample(n=sample_size, random_state=2024)
            
        augmented_rows = []
        
        with tqdm(total=len(df), desc="Augmenting dataset") as pbar:
            for idx, row in df.iterrows():
                try:
                    original_data = {
                        'paragraph': row['paragraph'],
                        'problems': eval(row['problems'])
                    }
                    
                    new_question = self.generate_augmented_question(original_data)
                    if new_question:
                        augmented_rows.append({
                            'id': f"{row['id']}_augmented",
                            'paragraph': new_question['paragraph'],
                            'problems': str(new_question['problems']),
                            'question_plus': new_question['question_plus']
                        })
                    else:
                        print(f"Failed to generate question for row {row['id']}")
                        
                except Exception as e:
                    print(f"Error processing row {row['id']}: {str(e)}")
                    continue
                
                pbar.update(1)
                if (pbar.n % 10 == 0):
                    print(f"\nTotal tokens used so far: {self.total_tokens:,}")
        
        print(f"\nFinal token usage: {self.total_tokens:,}")
        print(f"Estimated cost: ${self.total_tokens * 0.000001:.4f}")
        print(f"\nSuccessfully generated {len(augmented_rows)} new questions")
        
        if not augmented_rows:
            raise ValueError("No questions were successfully generated")
            
        return pd.DataFrame(augmented_rows)

if __name__ == "__main__":
    # 설정
    API_KEY = "YOUR_API_KEY"
    INPUT_FILE = "train.csv"
    OUTPUT_FILE = "train_augmentation_1000.csv"
    SAMPLE_SIZE = 1000
    
    # 데이터 증강
    try:
        augmenter = QuestionAugmenter(API_KEY)
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} questions from {INPUT_FILE}")
        
        augmented_df = augmenter.augment_dataset(df, SAMPLE_SIZE)
        augmented_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAugmented dataset saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
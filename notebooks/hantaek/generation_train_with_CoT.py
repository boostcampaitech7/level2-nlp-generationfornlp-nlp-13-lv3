import openai
import pandas as pd
from typing import Dict
from tqdm import tqdm

class CoTGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.total_tokens = 0
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_cot",
                    "description": "문제 해결을 위한 Chain of Thought를 생성합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question_plus": {
                                "type": "string",
                                "description": "문제 해결을 위한 단계적 사고 과정"
                            }
                        },
                        "required": ["question_plus"],
                        "additionalProperties": False
                    }
                },
                "strict": True
            }
        ]

        # Few-shot 예시를 포함한 시스템 프롬프트
        self.system_prompt = """당신은 대한민국 수능 문제 분석 전문가입니다. 
다음 예시들을 참고하여 주어진 문제에 대한 단계적 사고 과정(Chain of Thought)을 생성해주세요.

예시 1)
[문제]
지문: 1960년대 이후 한국 경제는 정부 주도의 경제 개발 계획에 따라 급속한 성장을 이루었다. 정부는 전략 산업을 선정하여 집중 육성하였고, 수출 주도형 정책을 시행하였다.
질문: 이 시기 경제 성장의 특징으로 가장 적절한 것은?
선택지: ['자유 방임 경제', '중화학 공업 육성', '내수 시장 활성화', '균형 발전']
정답: 2

[사고 과정]
1. 시대적 배경 파악
   - 1960년대 이후 한국의 경제 개발 시기
   - 정부 주도의 계획 경제 체제

2. 핵심 정책 분석
   - 전략 산업 집중 육성
   - 수출 주도형 정책 시행

3. 선택지 검토
   - 자유 방임 경제: 정부 주도 정책과 상반됨
   - 중화학 공업 육성: 당시 대표적인 전략 산업
   - 내수 시장 활성화: 수출 중심 정책과 맞지 않음
   - 균형 발전: 선별적 육성 정책과 상반됨

4. 정답 도출
   중화학 공업 육성이 정부의 전략 산업 정책과 일치

예시 2)
[문제]
지문: 신라는 삼국 통일 이후 귀족들의 경제적 기반이 확대되었고, 그들의 호화로운 생활이 사회적 문제가 되었다.
질문: 이 시기 신라의 모습으로 적절한 것은?
선택지: ['녹읍이 부활되었다', '골품제가 폐지되었다', '화백 회의가 폐지되었다', '집사부 체제가 도입되었다', '농민의 세력이 강화됐다']
정답: 1

[사고 과정]
1. 시기 확인
   - 신라 삼국 통일 이후
   - 귀족 세력 강화 시기

2. 핵심 상황 분석
   - 귀족의 경제적 기반 확대
   - 사치스러운 생활 문제화
   - 귀족 세력 강화 필요성

3. 관련 제도 검토
   - 녹읍: 귀족에게 토지와 노동력 지급
   - 골품제: 신분 제도로 이미 시행 중
   - 화백회의: 귀족 회의체로 유지
   - 집사부: 통일 이전 설치

4. 정답 도출
   귀족 세력 강화를 위해 녹읍 부활이 가장 적절

위 예시들을 참고하여 다음 형식으로 사고 과정을 작성해주세요:
1. 시대/상황/배경 파악
2. 핵심 개념/정책/제도 분석
3. 선택지 검토 및 분석
4. 오답인 이유를 구체적으로 제시
5. 정답 도출 근거 제시
6. 앞으로 어떤 수능 문제에 적용할 수 있을지 명확히 기술"""

    def generate_cot(self, question_data: Dict) -> str:
        """Chain of Thought 생성"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""다음 문제에 대한 Chain of Thought를 핵심만 요약해서 공백포함 500자 내외로 생성해주세요.
                    
문제 정보:
지문: {question_data['paragraph']}
질문: {question_data['problems']['question']}
선택지: {question_data['problems']['choices']}
정답: {question_data['problems']['answer']}

예시와 같은 형식으로 단계적인 사고 과정을 생성해주세요. Let's step by step!"""}
                ],
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "generate_cot"}},
                temperature=0.7,
                max_tokens=3000
            )
            
            if 'usage' in response:
                print(f"\nToken usage for this request:")
                print(f"Prompt tokens: {response['usage']['prompt_tokens']}")
                print(f"Completion tokens: {response['usage']['completion_tokens']}")
                print(f"Total tokens: {response['usage']['total_tokens']}")
                self.total_tokens += response['usage']['total_tokens']

            if hasattr(response.choices[0].message, 'tool_calls'):
                cot_response = eval(response.choices[0].message.tool_calls[0].function.arguments)
                return cot_response['question_plus']
            else:
                print("No tool calls in response")
                return None
            
        except Exception as e:
            print(f"\nError in generate_cot: {str(e)}")
            return None

    def update_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터셋의 question_plus 열 업데이트"""
        
        updated_df = df.copy()
        
        with tqdm(total=len(df), desc="Generating CoT") as pbar:
            for idx, row in df.iterrows():
                try:
                    question_data = {
                        'paragraph': row['paragraph'],
                        'problems': eval(row['problems'])
                    }
                    
                    new_cot = self.generate_cot(question_data)
                    if new_cot:
                        updated_df.at[idx, 'question_plus'] = new_cot
                    else:
                        print(f"Failed to generate CoT for row {row['id']}")
                        
                except Exception as e:
                    print(f"Error processing row {row['id']}: {str(e)}")
                    continue
                
                pbar.update(1)
                if (pbar.n % 10 == 0):
                    print(f"\nTotal tokens used so far: {self.total_tokens:,}")
        
        print(f"\nFinal token usage: {self.total_tokens:,}")
        print(f"Estimated cost: ${self.total_tokens * 0.0000003:.4f}")
        print(f"\nSuccessfully updated {len(df)} questions")
        
        return updated_df

if __name__ == "__main__":
    # 설정
    API_KEY = "YOUR_API_KEYS"
    INPUT_FILE = "train.csv"
    OUTPUT_FILE = "train_with_cot.csv"
    
    # CoT 생성 및 업데이트
    try:
        generator = CoTGenerator(API_KEY)
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} questions from {INPUT_FILE}")
        
        updated_df = generator.update_dataset(df)
        updated_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nUpdated dataset saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
import pandas as pd
from collections import Counter
import numpy as np

def ensemble_predictions(csv_files, method='majority', weights=None):
    """
    여러 예측 결과를 앙상블하는 함수
    
    Args:
        csv_files (list): CSV 파일 경로 리스트
        method (str): 'majority' (다수결) 또는 'weighted' (가중치 투표)
        weights (list): method가 'weighted'일 때 사용할 가중치 리스트
    """
    # 모든 예측 결과 로드
    predictions = []
    for file in csv_files:
        df = pd.read_csv(file)
        predictions.append(df.set_index('id')['answer'])
    
    # 결과를 저장할 DataFrame 생성
    result_df = pd.DataFrame(index=predictions[0].index)
    
    if method == 'majority':
        # 다수결 투표
        for idx in result_df.index:
            votes = [pred[idx] for pred in predictions]
            result_df.loc[idx, 'answer'] = Counter(votes).most_common(1)[0][0]
            
    elif method == 'weighted':
        if weights is None:
            weights = [1/len(predictions)] * len(predictions)
        # 가중치 투표
        for idx in result_df.index:
            votes = [pred[idx] for pred in predictions]
            vote_counts = Counter(votes)
            weighted_votes = {k: 0 for k in set(votes)}
            for vote, weight in zip(votes, weights):
                weighted_votes[vote] += weight
            # 동점일 경우 낮은 번호가 우선순위
            result_df.loc[idx, 'answer'] = max(weighted_votes.items(), key=lambda x: x[1])[0]
            # 동점일 경우 높은 번호가 우선순위
            # result_df.loc[idx, 'answer'] = max(weighted_votes.items(), key=lambda x: (x[1], x[0]))[0]
    
    # 신뢰도 점수 추가
    result_df['confidence'] = 0.0
    for idx in result_df.index:
        votes = [pred[idx] for pred in predictions]
        majority_count = Counter(votes).most_common(1)[0][1]
        result_df.loc[idx, 'confidence'] = majority_count / len(predictions)
    
    return result_df.reset_index()

def analyze_ensemble_results(result_df, predictions_dfs):
    """앙상블 결과 분석"""
    print("\nEnsemble Analysis:")
    print(f"Total samples: {len(result_df)}")
    
    # 신뢰도 분포
    print("\nConfidence Distribution:")
    print(result_df['confidence'].describe())
    
    # 모델 간 일치도
    print("\nModel Agreement Analysis:")
    agreement_counts = result_df['confidence'].value_counts().sort_index()
    for conf, count in agreement_counts.items():
        print(f"Agreement {conf*100:.0f}%: {count} samples")

def main():
    # CSV 파일 목록
    csv_files = [
        'output_0.8180_majority.csv',
        'output_0.7834_qnq.csv',
        'output_0.7488_electra.csv',
        'output_0.7488_s.csv',
        'output_0.7028_base.csv',
        'output_0.8111_exp2_weighted.csv'
        
    ]
    
    # 가중치 설정 (선택적)
    # 예: 성능이 더 좋은 모델에 더 높은 가중치
    weights = [0.3, 0.2, 0.2, 0.2, 0.1, 0.1] 
    
    # 다수결 방식 앙상블
    majority_results = ensemble_predictions(csv_files, method='majority')
    
    # 가중치 방식 앙상블
    weighted_results = ensemble_predictions(csv_files, method='weighted', weights=weights)
    
    # 결과 저장
    majority_results.to_csv('ensemble_majority.csv', index=False)
    weighted_results.to_csv('ensemble_weighted.csv', index=False)
    
    # 결과 분석
    predictions_dfs = [pd.read_csv(f) for f in csv_files]
    analyze_ensemble_results(majority_results, predictions_dfs)

def process_final_submission(input_file: str, output_file: str):
    """
    앙상블 결과를 최종 제출 형식으로 가공
    
    Args:
        input_file (str): 앙상블 결과 CSV 파일 경로 (ensemble_majority.csv 또는 ensemble_weighted.csv)
        output_file (str): 최종 제출용 CSV 파일 경로
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(input_file)
        
        # confidence 열 제거 및 answer를 int로 변환
        final_df = df[['id', 'answer']].copy()
        final_df['answer'] = final_df['answer'].astype(int)
        
        # 결과 저장
        final_df.to_csv(output_file, index=False)
        print(f"\nProcessed results saved to: {output_file}")
        
        # 결과 확인
        print("\nSample of processed results:")
        print(final_df.head())
        print("\nValue counts for answers:")
        print(final_df['answer'].value_counts().sort_index())
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    # 메인 앙상블 수행
    main()
    print("\n=== Processing Ensemble Results ===")
    
    # 다수결 방식 결과 처리
    process_final_submission(
        input_file='ensemble_majority.csv',
        output_file='ensemble_majority_output.csv'
    )
    
    # 가중치 방식 결과 처리
    process_final_submission(
        input_file='ensemble_weighted.csv',
        output_file='ensemble_weighted_output.csv'
    )
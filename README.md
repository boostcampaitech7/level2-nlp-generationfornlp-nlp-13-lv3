# Generation For NLP 프로젝트

## 개요
'한국어'와 '시험'이라는 주제에 맞춰서 작은 모델들로 수능 시험을 풀어보는 프로젝트

### 폴더 구조 설명
```sh
project-name/
│
├── data/                     # 데이터셋 폴더
│   
├── notebooks/                # 개인용 작업장(프로젝트 참여자가 하고 싶은 실험 프로토타입)
│
├── src/                      # 소스 코드
│   ├── data/                 # 데이터 로드, 저장 및 처리 코드
│   ├── models/               # 모델 정의 및 구조 코드
│   ├── training/             # 학습 루프, 손실 함수, 최적화 관련 코드
│   ├── evaluation/           # 모델 평가 코드 (메트릭 계산 등)
│   ├── utils/                # 보조 함수나 유틸리티 코드
│   └── visualization/        # 시각화 코드
│
├── experiments/              # 실험 관리 폴더(checkpoint 등)
│   
│
├── scripts/                  # 실행 가능한 스크립트 (주로 파이썬 진입점)
│   ├── train.py              # 학습 스크립트
│   ├── evaluate.py           # 평가 스크립트
│   └── predict.py            # 예측 스크립트
│
├── run                       # 콘솔 실행 자동화 스크립트
|   ├── run.sh          
├── config/                   # 설정 파일 (하이퍼파라미터 및 경로 설정)
│
│
├── requirements.txt          # 필요한 Python 패키지 목록
├── README.md                 # 프로젝트 개요 및 설명
└── .gitignore                # Git에서 제외할 파일 목록
```
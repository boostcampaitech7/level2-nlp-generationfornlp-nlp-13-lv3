#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN


NOHUP="False"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nohup) NOHUP="True"; shift ;;  
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
    shift
done

COMMAND="python scripts/predict_qwen_best.py --config_path ./config/qwen2.5_32B_unsloth_best_config.yaml"

if [[ "$NOHUP" == "True" ]]; then
    echo "nohup 모드로 실행합니다."
    nohup $COMMAND > ./logs/nohup_qwen_best_inference.log 2> ./logs/nohup_qwen_best_inference_error.log &
    echo "nohup 실행 완료. 로그: nohup_qwen_best.log, 에러: nohup_qwen_best_error.log"
else
    echo "일반 모드로 실행합니다."
    $COMMAND
fi
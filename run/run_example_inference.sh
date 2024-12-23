#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN

NOHUP="False"
SCRIPT_NAME="predict"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nohup) NOHUP="True"; shift ;;  
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
    shift
done

COMMAND="python scripts/${SCRIPT_NAME}.py --config_path ./config/ko_gemma_2B_config.yaml --checkpoint_path ./experiments/gemma-ko-2b-3ep/checkpoint-1125"


if [[ "$NOHUP" == "True" ]]; then
    echo "nohup 모드로 실행합니다."
    nohup $COMMAND > ./logs/${SCRIPT_NAME}.log 2> ./logs/${SCRIPT_NAME}_error.log &
    echo "nohup 실행 완료. 로그: ${SCRIPT_NAME}.log, 에러: ${SCRIPT_NAME}_error.log"
else
    echo "일반 모드로 실행합니다."
    $COMMAND
fi
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN
python scripts/predict.py --config_path ./config/ko_gemma_2B_config.yaml --checkpoint_path ./experiments/gemma-ko-2b-3ep/checkpoint-1125
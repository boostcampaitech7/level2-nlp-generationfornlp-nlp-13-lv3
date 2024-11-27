#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN
python scripts/train_qwen_best.py --config_path ./config/qwen2.5_32B_unsloth_best_config.yaml --is_augmented True
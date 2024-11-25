#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN
python scripts/train_dataset_test.py --config_path ./config/qwen2.5_32B_unsloth_dataset_test_config_ver2.yaml
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN
python scripts/train.py --config_path ./config/ko_gemma_2B_config.yaml
import os
import torch
import numpy as np
import random


# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_latest_checkpoint(output_dir):
    """
    output 디렉터리에서 'checkpoint-숫자' 형식의 폴더 중 가장 높은 숫자의 폴더명을 반환하는 함수

    Args:
        output_dir (str): output directory path

    Returns:
        str: 체크포인트 중 가장 높은 숫자의 숫자를 반환
    """

    checkpoint_folders = [folder for folder in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, folder)) and folder.startswith("checkpoint-")]

    checkpoint_numbers = [int(folder.split("-")[1]) for folder in checkpoint_folders if folder.split("-")[1].isdigit()]

    if checkpoint_numbers:
        latest_checkpoint_number = max(checkpoint_numbers)
        return f"{output_dir}/checkpoint-{latest_checkpoint_number}"
    else:
        return None

"""
unsloth 라이브러리의 현재 시스템에서 사용 가능한 설치명령어를 문자열로 가져오는 코드입니다.
https://github.com/unslothai/unsloth/의 README.md에 작성되어있습니다.

"""

try:
    import torch
except:
    raise ImportError("Install torch via `pip install torch`")
from packaging.version import Version as V

v = V(torch.__version__)
cuda = str(torch.version.cuda)
is_ampere = torch.cuda.get_device_capability()[0] >= 8
if cuda != "12.1" and cuda != "11.8" and cuda != "12.4":
    raise RuntimeError(f"CUDA = {cuda} not supported!")
if v <= V("2.1.0"):
    raise RuntimeError(f"Torch = {v} too old!")
elif v <= V("2.1.1"):
    x = "cu{}{}-torch211"
elif v <= V("2.1.2"):
    x = "cu{}{}-torch212"
elif v < V("2.3.0"):
    x = "cu{}{}-torch220"
elif v < V("2.4.0"):
    x = "cu{}{}-torch230"
elif v < V("2.5.0"):
    x = "cu{}{}-torch240"
elif v < V("2.6.0"):
    x = "cu{}{}-torch250"
else:
    raise RuntimeError(f"Torch = {v} too new!")
x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
print(f'pip install --upgrade pip && pip install "unsloth[{x}] @ git+https://github.com/unslothai/unsloth.git"')

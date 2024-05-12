import os
import cv2
import torch
import random
import numpy as np
from typing import Dict, Any, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    import yaml

    # YAML 파일 경로
    config_path = "config.yaml"

    # YAML 파일 읽기
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg


def seed_everything(seed: int = 42):
    """
        Seed Everything
    Python과 PyTorch를 사용하여 모든 난수 생성을 일관되게 만드는 함수
    모델의 재현성을 보장하고, 실험의 결과를 일관되게 만들기 위해 사용
    """

    # Python의 내장 random 모듈의 난수 생성을 seed 값에 따라 초기화합니다.
    # 이를 통해 random 모듈에서 생성되는 모든 난수는 일관되게 됩니다.
    random.seed(seed)

    # 운영 체제의 환경 변수 PYTHONHASHSEED를 설정하여 Python의 해시 함수에 사용되는 난수 생성을 일관되게 만듭니다.
    # Python 3.6 이상에서 사용 가능합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy의 난수 생성을 seed 값에 따라 초기화합니다.
    # 이를 통해 NumPy에서 생성되는 모든 난수는 일관되게 됩니다.
    np.random.seed(seed)

    # PyTorch의 난수 생성을 seed 값에 따라 초기화합니다.
    # 이를 통해 PyTorch에서 생성되는 모든 난수는 일관되게 됩니다.
    torch.manual_seed(seed)

    # GPU에서 PyTorch의 난수 생성을 seed 값에 따라 초기화합니다.
    # 이는 GPU에서 실행되는 코드의 결과를 일관되게 만듭니다.
    torch.cuda.manual_seed(seed)

    # PyTorch의 cuDNN 백엔드를 설정하여, GPU에서 실행되는 연산의 결과가 항상 동일하게 됩니다.
    # 이는 모델의 재현성을 보장하는 데 중요합니다.
    torch.backends.cudnn.deterministic = True

    # PyTorch의 cuDNN 백엔드를 설정하여, GPU에서 실행되는 연산의 최적화가 가능하도록 합니다.
    # 이는 모델의 성능을 향상시키는 데 도움이 됩니다.
    torch.backends.cudnn.benchmark = True


def get_img(path: str) -> np.array:
    """
    이미지 파일을 읽어들이는 함수
    """

    # 이미지 파일을 읽어들이는 코드
    img_bgr = cv2.imread(path)
    img_rgb = img_bgr[:, :, ::-1]
    return img_rgb


def rand_bbox(size: Tuple, lam):
    """
    주어진 이미지 크기와 랜덤하게 생성된 랜덤 램프의 크기와 위치를 반환하는 함수입니다.
    이 함수는 이미지에 랜덤한 램프를 적용할 때 사용될 수 있습니다.
    램프는 이미지의 일부를 랜덤하게 잘라내는 데 사용되는 기법입니다.

    size: 이미지의 크기를 나타내는 튜플 (W, H)
    lam: 램프의 크기를 결정하는 파라미터. 이 값이 클수록 램프의 크기가 작아집니다.
    """
    W, H = size[0:2]  # 이미지의 너비와 높이를 추출합니다.
    cut_rat = np.sqrt(1.0 - lam)  # 램프의 크기를 결정하는 비율을 계산합니다.
    cut_w = np.int(W * cut_rat)  # 램프의 너비를 계산합니다.
    cut_h = np.int(H * cut_rat)  # 램프의 높이를 계산합니다.

    # 램프의 중심점을 랜덤하게 선택합니다.
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 램프의 왼쪽 상단 모서리와 오른쪽 하단 모서리를 계산합니다.
    # array 내의 element들에 대해서 min 값 보다 작은 값들을 min값으로 바꿔주고
    # max 값 보다 큰 값들을 max값으로 바꿔주는 함수.
    bbx1 = np.clip(a=cx - cut_w // 2, a_min=0, a_max=W)
    bby1 = np.clip(a=cy - cut_h // 2, a_min=0, a_max=H)
    bbx2 = np.clip(a=cx + cut_w // 2, a_min=0, a_max=W)
    bby2 = np.clip(a=cy + cut_h // 2, a_min=0, a_max=H)

    # 램프의 위치와 크기를 반환합니다.
    return bbx1, bby1, bbx2, bby2

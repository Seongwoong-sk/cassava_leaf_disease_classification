import torch
import numpy as np
import pandas as pd
from typing import Any, Dict
from torch.utils.data import Dataset
from utils.helper import load_config, get_img, rand_bbox
from utils.fmix import sample_mask, make_low_freq_image, binarise_mask


class CassavaDataset(Dataset):
    """
    CassavaDataset 클래스는 PyTorch의 Dataset 클래스를 상속받아, 카사바 잎 질병 분류에 사용되는 데이터셋을 처리하는 역할을 합니다.
    이 클래스는 데이터셋의 이미지와 레이블을 로드하고, 필요한 경우 데이터 증강을 적용합니다.
    데이터 증강에는 FMix와 CutMix가 포함됩니다.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transforms: bool = None,
        output_label: bool = True,
        one_hot_label: bool = False,
        do_fmix: bool = False,
        cfg_path: bool = None,
        do_cutmix: bool = False,
        cutmix_params: Dict = {"alpha": 1},
    ) -> None:
        # 설정 파일 로드
        if cfg_path:
            self._cfg = load_config(cfg_path=cfg_path)
        else:
            self._cfg = {}

        # fmix_params를 cfg에서 가져오거나 기본값으로 설정
        # FMix 파라미터 설정
        self._fmix_params = {
            "alpha": 1.0,
            "decay_power": 3.0,
            "shape": (self._cfg.get("img_size"), self._cfg.get("img_size")),
        }

        # NOTE
        # 이는 CassavaDataset 클정래스가 Dataset 클래스의 초기화 로직을 사용하고 싶을 때 필요합니다.
        # 예를 들어, Dataset 클래스는 데이터셋을 로드하거나 데이터를 준비하는 로직을 포함하고 있을 수 있으며, 이를 CassavaDataset 클래스에서도 재사용하고 싶을 수 있습니다.
        super().__init__()  # 현재 클래스가 상속받은 부모 클래스의 __init__ 메소드를 호출

        # 데이터 프레임 복사
        self._df = df.reset_index(drop=True).copy()
        self._transforms = transforms
        self._data_root = data_root
        self._do_fmix = do_fmix
        self._do_cutmix = do_cutmix
        self._cutmix_params = cutmix_params

        self._output_label = output_label
        self._one_hot_label = one_hot_label

        # 레이블 출력 여부와 one-hot 인코딩 여부 설정
        if output_label == True:
            self._labels = self.df["label"].values

            if one_hot_label is True:
                self._labels = np.eye(self.df["label"].max() + 1)[self._labels]

    def __len__(self) -> int:
        """데이터셋의 총 샘플 수를 반환"""
        return self.df.shape[0]

    def __getitem__(self, index) -> Any:
        """
        주어진 인덱스에 해당하는 샘플의 이미지와 레이블을 반환합니다
        레이블은 output_label이 True일 때만 반환됩니다.
        또한, FMix와 CutMix 데이터 증강을 적용할 수 있습니다.
        """

        # 레이블 로드
        if self._output_label:
            target = self._labels[index]

        # 이미지 로드
        img = get_img("{}/{}".format(self._data_root, self._df.loc[index]["image_id"]))

        # 데이터 변환 적용
        if self._transforms:
            img = self._transforms(image=img)["image"]

        # FMix 데이터 증강 적용
        if self._do_fmix and np.random.uniform(0.0, 1.0, size=1)[0] > 0.5:
            with torch.no_grad():
                # lam, mask = sample_mask(**self._fmix_params)

                lam = np.clip(
                    a=np.random.beta(
                        self._fmix_params["alpha"], self._fmix_params["alpha"]
                    ),
                    a_min=0.6,
                    a_max=0.7,
                )

                # Make mask, get mean / std
                mask = make_low_freq_image(
                    decay=self._fmix_params["decay_power"],
                    shape=self._fmix_params["shape"],
                )
                mask = binarise_mask(
                    mask=mask,
                    lam=lam,
                    in_shape=self._fmix_params["shape"],
                    max_soft=self._fmix_params["max_soft"],
                )
                fmix_ix = np.random.choice(self._df.index, size=1)[0]
                fmix_img = get_img(
                    "{}/{}".format(self._data_root, self._df.iloc[fmix_ix]["image_id"])
                )

                if self._transforms:
                    fmix_img = self._transforms(image=fmix_img)["image"]

                mask_torch = torch.from_numpy(ndarray=mask)
                # 이미지 믹스
                img = mask_torch * img + (1.0 - mask_torch) * fmix_img

                # assert img.shape == self스._fmix_params['shape']

                # 레이블 믹스
                rate = mask.sum() / self._cfg["img_size"] / self._cfg["img_size"]
                target = rate * target + (1.0 - rate) * self._labels[fmix_ix]
                # print(target, mask, img)
                # assert False

        # CutMix 데이터 증강 적용
        if self._do_cutmix and np.random.uniform(0.0, 1.0, size=1)[0] > 0.5:
            # print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(a=self._df.index, size=1)[0]
                cmix_img = get_img(
                    "{}/{}".format(self._data_root, self._df.iloc[cmix_ix]["image_id"])
                )

                if self._transforms:
                    cmix_img = self._transforms(image=cmix_img)["image"]

                lam = np.clip(
                    a=np.random.beta(
                        self._cutmix_params["alpha"], self._cutmix_params["alpha"]
                    ),
                    a_min=0.3,
                    a_max=0.4,
                )

                bbx1, bby1, bbx2, bby2 = rand_bbox(
                    size=(self._cfg["img_size"], self._cfg["img_size"]), lam=lam
                )

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - (
                    (bbx2 - bbx1)
                    * (bby2 - bby1)
                    / (self._cfg["img_size"] * self._cfg["img_size"])
                )

                target = rate * target + (1.0 - rate) * self._labels[cmix_ix]

            # print('-', img.sum())
            # print(target)
            # assert False

        # 레이블 스무딩
        if self._output_label == True:
            return img, target
        else:
            return img

import timm
import torch.nn as nn
from torch import Tensor

class CassavaImgClassifier(nn.Module):
    def __init__(self, model_arch: str, 
                 n_class: int, 
                 pretrained: bool =False) -> None:
        super.__init__()
        self._model = timm.create_model(model_name=model_arch, pretrained=pretrained)
        n_features = self._model.classifier.in_features
        self._model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
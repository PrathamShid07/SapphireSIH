from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
from .uavid_classes import NUM_CLASSES


_MODEL_REGISTRY = {
    "deeplabv3_mobilenet_v3_large": (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    ),
}


class SegmentationModel(nn.Module):
    def __init__(self, model_name="deeplabv3_mobilenet_v3_large", num_classes=NUM_CLASSES):
        super().__init__()
        if model_name not in _MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY)}")

        factory, weights = _MODEL_REGISTRY[model_name]
        self._model = factory(weights=weights)

        in_channels = self._model.classifier[4].in_channels
        self._model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        if self._model.aux_classifier is not None:
            in_ch_aux = self._model.aux_classifier[4].in_channels
            self._model.aux_classifier[4] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    def forward(self, x):
        return self._model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            out = self._model(x)
        return out["out"].argmax(dim=1)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))

    def load(self, path):
        state = torch.load(str(path), map_location="cpu")
        self._model.load_state_dict(state)
        return self

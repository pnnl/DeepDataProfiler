import pytest
import torch
from torchvision.models import resnet18
import deep_data_profiler as ddp


class VisionExample:
    def __init__(self):
        torch.manual_seed(0)
        self.model = resnet18(pretrained=True).to("cpu").eval()


@pytest.fixture
def resnet18_example():
    return VisionExample()

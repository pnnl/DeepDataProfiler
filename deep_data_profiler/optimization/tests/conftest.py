import pytest
import torch
from torchvision.models import resnet18
import deep_data_profiler as ddp
from deep_data_profiler.optimization import project_svd


class VisionExample:
    def __init__(self):
        torch.manual_seed(0)
        self.model = resnet18(pretrained=True).to("cpu").eval()
        self.profiler = ddp.ChannelProfiler(self.model)
        self.svd_projection = project_svd(self.profiler)
        self.hooked_model = ddp.utils.TorchHook(self.model)


@pytest.fixture
def resnet18_example():
    return VisionExample()

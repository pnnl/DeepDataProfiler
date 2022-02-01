import pytest
import torch
from torchvision.models import resnet18
import deep_data_profiler as ddp
from deep_data_profiler.optimization import project_svd


class VisionExample:
    def __init__(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=True).to(self.device).eval()
        self.profiler = ddp.ChannelProfiler(self.model)
        self.svd_projection = project_svd(self.profiler, device=self.device)
        self.hooked_model = ddp.utils.TorchHook(self.model, device=self.device)


@pytest.fixture
def resnet18_example():
    return VisionExample()

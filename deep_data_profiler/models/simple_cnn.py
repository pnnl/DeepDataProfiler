import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(32 * 7 * 7, 784),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, num_classes))

    def forward(self, X):
        x = self.features(X)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
"""Small residual CNN with rotation/reflection averaging at inference."""
import torch
import torch.nn as nn

# Computed from training images only. Teacher's test.py applies these constants.
MEAN = [0.79434784, 0.65965901, 0.69619251]
STD = [0.21563033, 0.24160339, 0.11788897]


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
        )
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32), ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(128, 8))
        # Stored in state_dict so the validation-selected inference mode reloads.
        self.register_buffer('use_tta', torch.tensor(True))

    def forward_once(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))

    def forward(self, x):
        if self.training or not bool(self.use_tta):
            return self.forward_once(x)
        # Eight views share the same parameters; no extra model weights.
        scores = None
        for flip in (False, True):
            for rotation in range(4):
                view = x.flip(-1) if flip else x
                view = view.rot90(rotation, (-2, -1))
                prob = self.forward_once(view).softmax(1)
                scores = prob if scores is None else scores + prob
        return (scores / 8).clamp_min(1e-8).log()

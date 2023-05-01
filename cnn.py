import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.featureMaps = nn.ModuleList()
        self.featureMaps.append(
            nn.Conv2d(1, 100, (3, args.kdim), device=0)
        )
        self.featureMaps.append(
            nn.Conv2d(1, 100, (4, args.kdim), device=0)
        )
        self.featureMaps.append(
            nn.Conv2d(1, 100, (5, args.kdim), device=0)
        )
        self.fullyConnectedLayer = nn.Linear(300, 2)
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        # self.featureMaps1 = nn.Conv2d(1, 100, (1, args.kdim), device=0)
        # self.featureMaps2 = nn.Conv2d(1, 100, (2, args.kdim), device=0)
        self.featureMaps3 = nn.Conv2d(1, 100, (3, args.kdim), device=0)
        self.featureMaps4 = nn.Conv2d(1, 100, (4, args.kdim), device=0)
        self.featureMaps5 = nn.Conv2d(1, 100, (5, args.kdim), device=0)
        self.dropout = nn.Dropout(p=args.dropout)
        self.dropoutRate = args.dropout
        self.fullyConnectedLayer = nn.Linear(300, 2, device=0)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, stage):
        x = torch.unsqueeze(x, 0)
        # x1 = self.featureMaps1(x)
        # x2 = self.featureMaps2(x)
        x3 = self.featureMaps3(x)
        x4 = self.featureMaps4(x)
        x5 = self.featureMaps5(x)
        # x1, _indices1 = torch.max(x1, 1)
        # x2, _indices2 = torch.max(x2, 1)
        x3, _indices3 = torch.max(x3, 1)
        x4, _indices4 = torch.max(x4, 1)
        x5, _indices5 = torch.max(x5, 1)
        x = torch.cat((x3, x4, x5), 0)
        x = torch.swapaxes(x, 0, 1)
        if stage == "train":
            x = self.dropout(x)
        else:
            x *= self.dropoutRate
        x = self.fullyConnectedLayer(x)
        x = self.softmax(x)
        return x[0]
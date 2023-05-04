import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.featureMaps3 = nn.Conv2d(1, args.cnn_kernel_num, (3, args.kdim), device=0)
        self.featureMaps4 = nn.Conv2d(1, args.cnn_kernel_num, (4, args.kdim), device=0)
        self.featureMaps5 = nn.Conv2d(1, args.cnn_kernel_num, (5, args.kdim), device=0)
        self.dropout = nn.Dropout(p=args.dropout)
        self.dropoutRate = 1 - args.dropout
        self.fullyConnectedLayer = nn.Linear(args.cnn_kernel_num * 3, 2, device=0)

    def forward(self, x, stage):
        x = torch.unsqueeze(x, 0)
        x3 = self.featureMaps3(x)
        x4 = self.featureMaps4(x)
        x5 = self.featureMaps5(x)
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
        return x[0]
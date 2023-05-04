import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.h0 = nn.Parameter(torch.randn(2, args.rnn_hdim, device=0))
        self.c0 = nn.Parameter(torch.randn(2, args.rnn_hdim, device=0))
        self.lstm = nn.LSTM(args.kdim, args.rnn_hdim, 2, device=0)
        self.dropout = nn.Dropout(p=args.dropout)
        self.dropoutRate = 1 - args.dropout
        self.fullConnection = nn.Linear(args.rnn_hdim * 2, 2, device=0)

    def forward(self, x, stage):
        output, (x, cn) = self.lstm(x, (self.h0, self.c0))
        x = x.view(1, -1)
        if stage == "train":
            x = self.dropout(x)
        else:
            x *= self.dropoutRate
        x = self.fullConnection(x)
        return x[0]
    
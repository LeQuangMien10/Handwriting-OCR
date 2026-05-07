### Pipeline: Image → CNN Backbone → Feature map
### → Convert to sequence → BiLSTM → Character probabilities
### → CTC Loss

import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, 2, padding=1),
            nn.ReLU(),
        )

        # BiLSTM
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 512, 256),
            BidirectionalLSTM(256, 256, num_classes)
        )

    def forward(self, x):
        # CNN
        print("Input:", x.shape)
        conv = self.cnn(x)
        print("CNN output:", conv.shape)

        # shape: (B, C, H, W)
        b, c, h, w = conv.size()

        assert h == 1, f"Expected height=1, got {h}"

        conv = conv.squeeze(2)

        conv = conv.permute(2, 0, 1)

        output = self.rnn(conv)

        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()

        self.rnn = nn.LSTM(
            n_in,
            n_hidden,
            bidirectional=True,
        )

        self.fc = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.rnn(x)

        T, B, H = recurrent.size()

        recurrent = recurrent.view(T * B, H)

        output = self.fc(recurrent)

        output = output.view(T, B, -1)

        return output
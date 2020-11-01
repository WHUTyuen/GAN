import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),#14*14
            nn.Conv2d(128, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),#7*7
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*512,128),
            nn.Sigmoid()
        )
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = conv1.reshape(-1,7*7*512)
        out = self.fc(conv1)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128,7*7*512),
            nn.BatchNorm1d(7*7*512),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(512,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),#14*14
            nn.ConvTranspose2d(128, 1, 3, 2, 1, 1),
            nn.LeakyReLU()  # 28*28
        )
    def forward(self, x):
        fc = self.fc(x)
        fc = fc.reshape(-1,512,7,7)
        out = self.conv1(fc)
        return out

class Net_total(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        encoder = self.encoder(x)
        out = self.decoder(encoder)
        return out
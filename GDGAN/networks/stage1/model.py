import torch
import torch.nn as nn


class generator(nn.Module):
    def __init__(self, nc, input_dim=100, image_size=28, class_num=3):
        super(generator, self).__init__()
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(input_dim+class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * ((self.image_size // 4) ** 2)),
            nn.BatchNorm1d(128 * ((self.image_size // 4) ** 2)),
            nn.ReLU(),
        )
        print('fc')
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),
            nn.Sigmoid(),
        )
        print('deconv')
        torch.cuda.empty_cache()


    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    def __init__(self, nc, image_size, class_num):
        super(discriminator, self).__init__()
        self.image_size = image_size
        num = 128*((self.image_size // 4) ** 2)
        print(num)
        self.conv = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        print('conv')

        self.dc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        print('dc')
        self.cl = nn.Sequential(
            nn.Linear(1024, class_num),
        )
        print('cl')
        self.fc1 = nn.Sequential(
            nn.Linear(num, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        print('fc1')
        torch.cuda.empty_cache()


    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.image_size // 4) ** 2)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c

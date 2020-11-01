import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image
import os

class D_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.dnet(x)
        return out

class G_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnet = nn.Sequential(
            nn.Linear(128,256),
            nn.LeakyReLU(),
            nn.Linear(256,512),
            nn.LeakyReLU(),
            nn.Linear(512,784)
        )
    def forward(self, x):
        out = self.gnet(x)
        return out

if __name__ == '__main__':
    batch_size = 100
    num_epoch = 10
    if not  os.path.exists("img"):
        os.makedirs("img")

    mnist_data = datasets.MNIST("/data", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(mnist_data, batch_size, shuffle=True)

    d_net = D_Net().cuda()
    g_net = G_net().cuda()

    loss_fun = nn.BCELoss()
    d_opt = torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))
    g_opt = torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))

    for epoch in range(num_epoch):
        for i,(img,label) in enumerate(train_loader):
            real_img = img.reshape(-1,784).cuda()
            real_label = torch.ones(img.size(0),1).cuda()
            fake_label = torch.zeros(img.size(0),1).cuda()

            real_out = d_net(real_img)
            d_loss_real = loss_fun(real_out,real_label)

            z = torch.randn(img.size(0),128).cuda()
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_loss_fake = loss_fun(fake_out,fake_label)

            d_loss = d_loss_real+d_loss_fake
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            #训练生成器#
            z = torch.randn(img.size(0),128).cuda()
            fake_img = g_net(z)
            g_fake_out = d_net(fake_img)
            g_loss = loss_fun(g_fake_out,real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i%10 == 0:
                print("Epoch:{0},d_loss{1},g_loss{2}".format(epoch,d_loss,g_loss))
                real_img = real_img.reshape(-1,1,28,28)
                fake_img = fake_img.reshape(-1,1,28,28)
                save_image(real_img,"img/{}-real_img.jpg".format(epoch+1),nrow=10,normalize=True,scale_each=True)
                save_image(fake_img, "img/{}-fake_img.jpg".format(epoch + 1), nrow=10, normalize=True, scale_each=True)

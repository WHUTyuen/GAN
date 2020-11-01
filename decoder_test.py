import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image
import net
import os

if __name__ == '__main__':
    if not os.path.exists("img"):
        os.makedirs("img")
    tf = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist_data = datasets.MNIST("/data",train=True,transform=tf,download=True)
    train_loader = DataLoader(mnist_data,100,shuffle=True)

    net = net.Net_total().cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_fun = nn.MSELoss()
    k = 0
    for epoch in range(100):
        for i,(img,label) in enumerate(train_loader):
            img = img.cuda()
            out_img = net(img)
            loss = loss_fun(out_img,img)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%10 == 0:
                print(loss.item())
                fack_img = out_img.detach()
                save_image(fack_img,"img/{}-fack_img.png".format(k),nrow=10)
                save_image(img,"img/{}-real_image.png".format(k),nrow=10)
                k+=1
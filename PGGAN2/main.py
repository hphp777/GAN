import configuration
import dataset
import network
import training
import config
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


# Generator, Discriminator Initialization
GAN_model = network.ProGAN(configuration.hparams)
for step in range(9):
    print(f"Current image size: {4 * 2 ** step}")
    training.GAN_trainer(GAN_model, configuration.ngpu, iter_per_save=68, filepath="./Checkpoints/")
    GAN_model.add_depth()


checkpoint = './Checkpoints/ProGAN-Scale(8)-Img(1024)-Iter(544).pth'
GAN_model.load_model(checkpoint)
# GAN_model.eval()

for i in range(1):
    with torch.no_grad():
        latent = torch.randn(1, 512) ###
        img = GAN_model(latent).detach()
        save_image(img*0.5+0.5, f"CXR_PGGAN/GAN/img/img_{i}.png")
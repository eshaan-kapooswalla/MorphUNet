from torch.utils.data import DataLoader
from data_loader.data_loader import CustomImageDataset
import models.UNET as UNET
import models.UNet_3Plus
import torch.nn as nn
import torch.optim as op
import torch
from tqdm import tqdm

from display_helper import display_image

def date_report(data):
    image = data[0]
    data_min = torch.min(image)
    data_max = torch.max(image)
    data_mean = torch.mean(image)
    print(f'min: {data_min} max: {data_max} mean: {data_mean}')

TrainingCustomImageDataset = CustomImageDataset()
print(len(TrainingCustomImageDataset))
batch_size = 20

train_dataloader = DataLoader(TrainingCustomImageDataset, batch_size=batch_size, shuffle=True)

features = [12, 24, 48, 96, 192]
num_inp_channels = 3
num_labels = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

#model = UNET.UNET(features, num_inp_channels, num_labels).to(device)
model = models.UNet_3Plus.UNet_3Plus().to(device)


mse_loss = nn.MSELoss() #change to BCE
bce_loss = nn.CrossEntropyLoss()
optimizer = op.Adam(model.parameters(), lr=.01)
max_pool = nn.MaxPool2d(kernel_size=(4, 4))
epochs = 6000
display_every = 5
loss_list = []

for epoch in tqdm(range(epochs)):
    train_images, train_labels = next(iter(train_dataloader))
    train_labels = train_labels.to(device)/255
    train_images = train_images.to(device)/255

    train_labels = max_pool(train_labels.permute(0, 2, 1)).permute(0, 2, 1)
    train_images = max_pool(train_images.permute(0, 3, 2, 1))
    outs = model(train_images)

    outs = outs.permute(0, 3, 2, 1).squeeze()

    print(f'image data report')
    date_report(train_images)
    print(f'labels data report')
    date_report(train_labels)

    #loss = mse_loss(train_labels, outs)
    loss = bce_loss(train_labels, outs) might be messed up
    loss.backward()
    optimizer.step()

    loss_list.append(loss.detach().cpu())

    if epoch % display_every == 0:
        display_image(epoch, display_every, loss_list, outs, train_labels, train_images.permute(0, 3, 2, 1))
    print(f'epoch: {epoch} loss: {loss}')

print('DONE!')


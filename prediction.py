import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from load_data import mask
from torch.utils.data import DataLoader
import csv
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Net(Module):
    def __init__(self, num_class):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            # Linear(4 * 7 * 7, 4 * 7 * 7),
            # Linear(4 * 7 * 7, 4 * 7 * 7),
            Linear(128 * 7 * 7 * 2, num_class)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

test_db = mask('/home/xiejun/mask_PUF/data/demo_test/final', 224)
batchsz = 1000
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
device = torch.device('cpu')
outputpath = '1bot2bot.csv'
model = Net(num_class=2).to(device)
model.load_state_dict(torch.load('./models/best_bot12.mdl'))

def evalute(model, loader, outputpath):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for step, (x, y) in enumerate(loader):
        # if step>=1:
        #     break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():    #不需要计算梯度，所以加上不求导，验证集一定要加上这几句话
            logits = model(x)
            pred = logits.argmax(dim=1)
        with open(outputpath, 'a+') as f:
            csv_write = csv.writer(f)
            csv_write.writerows([logits.detach().numpy(), y.detach().numpy()])
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

test_acc = evalute(model, test_loader, outputpath)





import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from load_data import mask
from torch.utils.data import DataLoader
import csv
from torch.optim.lr_scheduler import StepLR

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class Resnet(nn.Module):
    # 224*224
    def __init__(self, block, num_layer, n_classes=2, input_channels=3):
        super(Resnet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(block.expansion * 512, block.expansion * 128)
        self.fc2 = nn.Linear(block.expansion * 128, block.expansion * 16)
        self.fc3 = nn.Linear(block.expansion * 16, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def resnet18(pretrained=False, **kwargs):
    model = Resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def main():
    history_path = './history.csv'
    num_class = 2
    model = resnet18().to(device)
    optimizer = SGD(model.parameters(), lr=lr,momentum=0.9, nesterov=True)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    criteon = CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        correct = 0
        total = len(train_loader.dataset)
        # scheduler.step()
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            # print(step)
            # if step>=2:
            #     break
            x, y = x.to(device), y.to(device)
            model.train()  # 必须加入model.train()，防止和验证时候BN一样
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # viz.line([loss.item()], [global_step], win='loss', update='append')  # loss可视化
            global_step += 1

        train_acc = correct / total
        print('train acc:', train_acc, 'epoch=', epoch)
        val_acc = evalute(model, val_loader)
        val_loss = evalute_loss(model, val_loader)
        print('val acc:', val_acc)
        print(loss.data.cpu().numpy())
        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.mdl')  # 保存模型权重
            # viz.line([val_acc], [global_step], win='val_acc', update='append')
            print('best acc:', best_acc, 'best epoch:', best_epoch)
            model.load_state_dict(torch.load('best.mdl'))  # 加载最好的模型权重
        # print('loaded from ckpt!')

            test_acc = evalute(model, test_loader)  # 验证模型，evalute需要我们自己写
            print('test acc:', test_acc)
        with open(history_path, "a+") as f:
            csv_write = csv.writer(f)
            csv_write.writerow([epoch, ';', train_acc, ';', val_acc, ';', loss.data.cpu().numpy(),';', val_loss.cpu().numpy()])

def evalute(model, loader):
    model.eval()   #必须要加入 model.eval() ，因为训练和测试BN不一致
    correct = 0
    total = len(loader.dataset)
    for step, (x, y) in enumerate(loader):
        # if step>=1:
        #     break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():    #不需要计算梯度，所以加上不求导，验证集一定要加上这几句话
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def evalute_loss(model, loader):
    model.eval()   #必须要加入 model.eval() ，因为训练和测试BN不一致
    correct = 0
    criteon = CrossEntropyLoss()
    total = len(loader.dataset)
    for step, (x, y) in enumerate(loader):
        # if step>=1:
        #     break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():    #不需要计算梯度，所以加上不求导，验证集一定要加上这几句话
            logits = model(x)
            loss = criteon(logits, y)
    return loss

batchsz = 128
lr = 1e-6
epochs = 500
device = torch.device('cuda')
torch.manual_seed(1234)

train_db = mask('/home/xiejun/mask_PUF/data/4up/train', 224)
val_db = mask('/home/xiejun/mask_PUF/data/4up/val', 224)
test_db = mask('/home/xiejun/mask_PUF/data/4up/test', 224)
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=0)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=0)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)

if __name__ == '__main__':
    main()

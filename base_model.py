import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from load_data import mask
from torch.utils.data import DataLoader
import csv

batchsz = 128
lr = 1e-6
epochs = 200
device = torch.device('cuda')
torch.manual_seed(1234)

train_db = mask('/home/xiejun/mask_PUF/data/1up2up/train', 224)
val_db = mask('/home/xiejun/mask_PUF/data/1up2up/val', 224)
test_db = mask('/home/xiejun/mask_PUF/data/1up2up/test', 224)
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=0)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=0)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)

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

def main():
    history_path = './mask_base.csv'
    num_class = 2
    model = Net(num_class=num_class).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criteon = CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        correct = 0
        total = len(train_loader.dataset)
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

        if epoch % 2 == 0:
            train_acc = correct / total
            print('train acc:', train_acc, 'epoch=', epoch)
            val_acc = evalute(model, val_loader)
            print('val acc:', val_acc)
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
                csv_write.writerow([epoch, ';', train_acc, ';', val_acc, ';', test_acc])

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

if __name__ == '__main__':
    main()

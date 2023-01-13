import csv
import glob
import random

import torch
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import os, sys

# class NumberDataset(Dataset):
#     def __init__(self, training=True):
#         if training:
#             self.samples = list(range(1, 1001))
#         else:
#             self.samples = list(range(1001, 15001))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         return self.samples(idx)
#
class mask(Dataset):
    def __init__(self, root, resize):
        super(mask, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv('images.csv')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([lambda x:Image.open(x).convert('RGB'),
                                 transforms.Resize(
                                     (int(self.resize * 1.25), int(self.resize *1.25))),
                                 transforms.RandomRotation(15),
                                 transforms.CenterCrop(self.resize),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                 ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.bmp'))
            print(len(images), images)
            random.shuffle(images)

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('written into csv file:', filename)

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

# db = mask('/home/xiejun/keras_resnet/dataset/comp/split1_pure', 64)
# x, y = next(iter(db))
# print('sample:', x.shape, y.shape, y)
# loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
# for x, y in loader: #此时x,y是批量的数据
#     print(len(x))
#     print(len(y))

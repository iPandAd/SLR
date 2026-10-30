import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import config
import torchvision.transforms as transforms


# 手语数据集
class SignMNIST(Dataset):
    def __init__(
        self, annotations_file,
        img_dir,
        transform=None,
        target_transform=None
    ):
        super(Dataset, self).__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# 数据集导入
train_dataset = SignMNIST(
    annotations_file="data/train.csv",
    img_dir="data/train/",
    transform=transforms.ToTensor()
)

test_dataset = SignMNIST(
    annotations_file="data/test.csv",
    img_dir="data/test/",
    transform=transforms.ToTensor()
)

# 数据集加载
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

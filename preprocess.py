import cv2
import numpy as np
import os
import pandas as pd


# 图像存储路径，若路径不存在则新建路径
TEST_PATH = "data/test/"
TRAIN_PATH = "data/train/"
CSV_PATH = "data/"

if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)
if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)
if not os.path.exists(CSV_PATH):
    os.makedirs(CSV_PATH)

# 创建标签与图像对应关系的容器
column_names = ["filename", "label"]
test_filename_label = []
train_filename_label = []

# 读取测试集与训练集CSV文件
test = pd.read_csv("data/sign_mnist_test.csv")
train = pd.read_csv("data/sign_mnist_train.csv")

# 将测试集特征转换为图像并进行相应记录
test_label = test.iloc[:, 0].values
test_features = test.iloc[:, 1:].values
test_width = int(np.log10(len(test))) + 1

for i in range(len(test)):
    test_img = test_features[i].reshape(28, 28).astype(np.uint8)
    test_filename = str(i).zfill(test_width) + ".jpg"
    test_filename_label.append([test_filename, test_label[i]])
    cv2.imwrite(TEST_PATH + test_filename, test_img)

# 将训练集特征转换为图像并进行相应记录
train_label = train.iloc[:, 0].values
train_features = train.iloc[:, 1:].values
train_width = int(np.log10(len(train))) + 1

for i in range(len(train)):
    train_img = train_features[i].reshape(28, 28).astype(np.uint8)
    train_filename = str(i).zfill(train_width) + ".jpg"
    train_filename_label.append([train_filename, train_label[i]])
    cv2.imwrite(TRAIN_PATH + train_filename, train_img)

# 将图像文件名与标签对应关系列表转为CSV文件
test_df = pd.DataFrame(test_filename_label, columns=column_names)
train_df = pd.DataFrame(train_filename_label, columns=column_names)

test_df.to_csv(CSV_PATH + "test.csv", index=False)
train_df.to_csv(CSV_PATH + "train.csv", index=False)

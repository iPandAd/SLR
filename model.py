from torch import relu, softmax
from torch.nn import Module, Conv2d, BatchNorm2d, Sequential, Linear
from torch.nn import functional as F


# 残差网络基本块
class ResBlock(Module):
    def __init__(self, ch_in, ch_out, stride=2):
        super(ResBlock, self).__init__()
        # 图像尺寸减半
        self.conv1 = Conv2d(
            ch_in, ch_out, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = BatchNorm2d(ch_out)
        # 图像尺寸不变
        self.conv2 = Conv2d(
            ch_out, ch_out, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = BatchNorm2d(ch_out)

        self.extra = Sequential(
            Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            BatchNorm2d(ch_out)
        )

    def forward(self, x):
        out = x
        x = relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        out = x + self.extra(out)
        return out


# ResNet18
class ResNet18(Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = Sequential(
            # 图像尺寸不变
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64)
        )
        # [b, 64, h, w] --> [b, 128, h, w]
        self.block1 = ResBlock(64, 128)
        # [b, 128, h, w] --> [b, 256, h, w]
        self.block2 = ResBlock(128, 256)
        # [b, 256, h, w] --> [b, 512, h, w]
        self.block3 = ResBlock(256, 512)
        # [b, 512, h, w] --> [b, 512, h, w]
        self.block4 = ResBlock(512, 512)
        # 全连接层，共24类
        self.outlayer = Linear(512, 26)

    def forward(self, x):
        x = relu(self.conv1(x))
        # [b, 64, h, w] --> [b, 1024, h, w]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # [b, 512, h, w] --> [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # 展开为列向量
        x = x.view(x.shape[0], -1)
        x = self.outlayer(x)
        x = softmax(x, dim=-1)
        return x

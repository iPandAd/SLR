# 配置文件，保存和管理项目的超参数、路径、模型配置
from torch import device
from torch.cuda import is_available


RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 5

NUM_FEATURES = 28 * 28
NUM_CLASSES = 24

# DEVICE = device("cuda" if is_available() else "cpu")
DEVICE = device("cpu")
GRAYSCALE = True

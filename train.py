import torch
import torch.nn as nn
import torch.optim
import config
import data_loader
from model import ResNet18


# 加载数据
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader

# 导入超参数
DEVICE = config.DEVICE
EPOCH = config.NUM_EPOCHS
print("[INFO] Process data on {}.".format(DEVICE))

# 模型、优化器及损失函数初始化
model = ResNet18()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

# 模型训练
print("Training process".center(80, "="))
for epoch in range(EPOCH):
    model.train()
    acc, nums = 0, 0
    for step, (b_x, b_y) in enumerate(train_loader):
        print("Epoch: {}, Step: {}".format(epoch, step))
        b_x = b_x.to(DEVICE)
        b_y = b_y.to(DEVICE)
        output = model(b_x)
        optimizer.zero_grad()
        loss = loss_func(output, b_y)
        loss.backward()
        optimizer.step()
        acc += sum(output.max(axis=1)[1] == b_y).cpu()
        nums += b_y.size()[0]
    print("[INFO] Epoch: {} | Loss: {:.6f} | Acc: {:.6f}".format(
        epoch, loss.item(), acc / nums))
    PATH = "checkpoints/resnet18_{}.tar".format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, PATH)
    print("[INFO] Save checkpoints at {}".format(PATH))
    print("".center(80, "-"))

# 模型评估
model.eval()
print("Evaluating process".center(80, "="))
for step, (b_x, b_y) in enumerate(test_loader):
    b_x = b_x.to(DEVICE)
    b_y = b_y.to(DEVICE)
    output = model(b_x)
    loss = loss_func(output, b_y)

    _, pred_y = torch.max(output.data, 1)
    correct = (pred_y == b_y).sum()
    total = b_y.size(0)
    loss = loss.cpu()
    correct = correct.cpu()

    print("[INFO] Epoch: {} | Test Loss: {:.6f} | Test Acc: {:.6f}".format(
        str(step).zfill(2), loss.data.numpy(), float(correct) / total
    ))

print("End".center(80, "="))

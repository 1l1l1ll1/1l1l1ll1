import numpy as np
import torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset,DataLoader
import torch


# 训练集
train_data = datasets.MNIST(root="./", # 存放位置
                            train = True, # 载入训练集
                            transform=transforms.ToTensor(), # 把数据变成tensor类型
                            download = True # 下载
                           )
# 测试集
test_data = datasets.MNIST(root="./",
                            train = False,
                            transform=transforms.ToTensor(),
                            download = True
                           )



# 批次大小
batch_size = 64
# 装载训练集
train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
# 装载测试集
test_loader = DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

for i,data in enumerate(train_loader):
    inputs,labels = data
    print(inputs.shape)
    print(labels.shape)
    break

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()# 初始化
        self.fc1 = nn.Linear(784,10)
        self.softmax = nn.Softmax(dim=1) # 激活函数 dim=1表示对第一个维度进行概率计算
        
    def forward(self,x):
        # torch.Size([64, 1, 28, 28]) -> (64,784)
        x = x.view(x.size()[0],-1) # 4维变2维 （在全连接层做计算只能2维）
        
        x = self.fc1(x) # 传给全连接层继续计算
        x = self.softmax(x) # 使用softmax激活函数进行计算
        return x
               

# 定义模型
model = Net()
# 定义代价函数
mse_loss = nn.CrossEntropyLoss()# 交叉熵
# 定义优化器
optimizer = optim.Adam(model.parameters(),lr=0.001)# Adam梯度下降

# 定义模型训练和测试的方法
def train():
    for i,data in enumerate(train_loader):
        # 获得一个批次的数据和标签
        inputs,labels = data
        out = model(inputs)
        # 交叉熵代价函数out（batch，C：类别的数量），labels（batch）
        loss = mse_loss(out,labels)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()
        
def test():
    correct = 0
    for i,data in enumerate(test_loader):
        # 获得一个批次的数据和标签
        inputs,labels = data
        # 获得模型预测结果
        out = model(inputs)
        # 获得最大值，以及最大值所在的位置
        _,predicted = torch.max(out,1)
        # 预测正确的数量
        correct += (predicted==labels).sum()
    print("Test acc:{0}".format(correct.item()/len(test_data)))

# 训练
for epoch in range(10):
    print("epoch:",epoch)
    train()
    test()
    print(model.parameters())

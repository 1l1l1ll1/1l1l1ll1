from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.w = torch.nn.Parameter(torch.randn(2,1)).cuda()
        self.b = torch.nn.Parameter(torch.randn(1)).cuda()
        
    def forward(self,x):
        print(self.w)
        print(self.b)
        y = (torch.matmul(self.w, x) + self.b).cuda()  # torch.t()  求矩阵的转置的函数
        return y

a = torch.rand(10,3)
b = torch.rand(10,1)
train_ids = TensorDataset(a, b)#封装数据a与标签b
# 切片输出
print(train_ids[0:2])
print('='* 80)
#  循环取数据
for x_train, y_label in train_ids:
     print(x_train, y_label)
# DataLoader进行数据封装
print('=' * 80)
 
train_loader = DataLoader(dataset=train_ids, batch_size=2, shuffle=False)
for i, data in enumerate(train_loader):  
    # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
    x_data, label = data
    print(' batch:{0}\n x_data:{1}\nlabel: {2}'.format(i, x_data, label))

model =Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
 
# Train the model
num_epochs= 20
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i ,data in enumerate(train_loader):

        # Forward pass
        x_data, label = data;
        outputs = model.forward(x_data)
        loss = criterion(outputs, label)
        print(loss)
 
 
        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
 
        if (i+1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    





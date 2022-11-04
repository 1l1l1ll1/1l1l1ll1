# 在pytorch中使用矩阵乘法实现全连接层
import torch.nn as nn
import torch
class FC(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(FC,self).__init__()
        self.w = torch.nn.Parameter(torch.randn(input_dim,output_dim))
        self.b = torch.nn.Parameter(torch.randn(output_dim))
        
    def forward(self,x):
        print("self.w  ",self.w)
        print("self.w.t()  ",self.w.t())
        y = torch.matmul(self.w.t(),x) + self.b  # torch.t()  求矩阵的转置的函数
        return y
linear = FC(3,2).cuda()
x = torch.ones(3).cuda()
y = linear(x)
print(y)
print(y.shape)
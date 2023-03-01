import torch
from torch.nn import functional as F

x = F.log_softmax(torch.randn(size=(16, 10)), dim=1)
y = torch.randint(0, 9, (16, ))
loss1 = F.nll_loss(x, target=y)

x1 = F.softmax(x,dim=1)
o1 = x1[range(16),y]
loss2 = torch.mean(-torch.log(o1))
print(loss1 == loss2)
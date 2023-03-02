import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torchvision.models import alexnet
from torch import autograd
model = alexnet()
criterion = CrossEntropyLoss(reduction="none")
x = torch.randn([16,3,227,227])
print(x.tanh(dim=1))

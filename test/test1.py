import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torchvision.models import alexnet
from torch import autograd
from torch.nn import LazyLinear, Linear

a = LazyLinear(8)
b = torch.randn((16, 9, 9))
print(a(b).shape)

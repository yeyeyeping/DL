## torch.randperm

torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor

返回从0到(n-1)数字的随机的排列

## torch.split

torch.split(tensor, split_size_or_sections, dim=0)

分割tensor成若干个块

tensor：torch.Tensor, 要分割的tensor

split_size_or_sections:{int, list(int)},单一chunk的数量

dim：int,沿着那个tensor进行分割




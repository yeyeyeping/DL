import torch
from torch import nn
import torch.nn.functional as F


def make_one_hot(label: torch.Tensor, classes_num: int):
    '''
    :param label:语义分割对应的标签，Bx1xHxW
    :param classes_num:类别数
    :return: 独热编码后的结果
    '''
    shape = (label.size(0), classes_num, label.size(2), label.size(3))
    zeros = torch.zeros(shape, device=label.device, dtype=torch.float)
    # scatter_将src(最后一个参数1) 维度1上按照index（label）填充到self(zeros)中
    target = zeros.scatter_(1, label.long(), 1)
    return target


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduce="mean", p=1):
        super().__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.p = p

    def forward(self, output, target):
        output_act = F.softmax(output, dim=1)
        target = make_one_hot(target.unsqueeze(1), output.size(1))
        # contiguous()用于判断self内部元素排列是否与当前的视图一致，不一致则创建
        out_flat, target_flat = output_act.contiguous().view(-1), \
            target.contiguous().view(-1)

        area_inter = (out_flat * target_flat).sum()
        area_union = (out_flat ** self.p + target_flat ** self.p).sum()
        loss = 1 - (2 * area_inter + self.smooth) / \
               (area_union + self.smooth)
        return loss


if __name__ == '__main__':
    label = torch.ones((16, 96, 96))
    output = torch.randn((16, 2, 96, 96))
    print(DiceLoss()(output, label))

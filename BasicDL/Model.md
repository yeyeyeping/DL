# 特征网络

## ResNet

![image-20230802135847116](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230802135847116.png)

## PPM模块、PSPNet( pyramid scene parsing network,2016)

![image-20230801144241158](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230801144241158.png)

对于backbone产生的feature，通过PPM进行不同尺度的池化

mmseg的实现十分优雅:

```python
class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

```

## FPN(Feature Pyramid Network,2017)

在FPN之前：[知乎](https://zhuanlan.zhihu.com/p/335333233)

![image-20230801145252542](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230801145252542.png)

![image-20230801145403276](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230801145403276.png)

FPN通常作为Neck部分，放在backbone之后，利用backbone卷积过程中，不同尺度的特征图，通过lateral connection产生多个尺度的输出

在做语义分割时，使用FPN Head产生预测结果，FPN Head将FPN传来的多尺度输出直接求和，然后通过convblock产生预测

## HRNet

![image-20230802113010665](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230802113010665.png)

保持高分辨率的同时、利用不同分支信息交互，补充通道数带来的信息损耗

hrnet主要由TransitionLayer（生成下采样分支）、FuseLayer（融合不同分支的信息）、Neck组成






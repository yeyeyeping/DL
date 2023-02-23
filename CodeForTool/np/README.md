## numpy基础

numpy默认创建的浮点类型为float64，注意这一torch tensor的默认行为不同，tensor默认情况下创建的float tensor是32位的，32位的浮点数已经能够满足梯度运算。因此在用numpy处理数据时需要注意转换成float32类型

## numpy api

### numpy.**pad**(*array*, *pad_width*, *mode='constant'*, ***kwargs*)

在卷积神经网络中，为了避免因为卷积运算导致输出图像缩小和图像边缘信息丢失，常常采用图像边缘填充技术，即在图像四周边缘填充0，使得卷积运算后图像大小不会缩小，同时也不会丢失边缘和角落的信息。

array：array_like,要填充的数组

pad_width: {sequence, array_like, int},填充的长度,可以是序列、数组、整数类型

如果是序列，必须是以下形式：

（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第n轴前面填充before_1个、后面填充after_1个数值。

mode: str,填充的方式，共计11种

![image-20230110112455354](README.assets/image-20230110112455354.png)


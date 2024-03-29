## numpy基础

numpy默认创建的浮点类型为float64，注意这一torch tensor的默认行为不同，tensor默认情况下创建的float tensor是32位的，32位的浮点数已经能够满足梯度运算。因此在用numpy处理数据时需要注意转换成float32类型

## numpy广播机制

广播机制首先需要判断参与计算的两个数组能否被广播机制处理？即判断是否广播兼容，规则是，比较两个数组的shape，`从shape的尾部开始向前一一比对`。

(1). 如果两个数组的`维度`相同，对应位置上轴的长度相同或其中一个的轴长度为1,广播兼容，可在轴长度为1的轴上进行广播机制处理。

(2). 如果两个数组的维度不同，那么给低维度的数组前扩展提升一维，扩展维的轴长度为1,然后在扩展出的维上进行广播机制处理。

## numpy api

### numpy.**pad**(*array*, *pad_width*, *mode='constant'*, ***kwargs*)

在卷积神经网络中，为了避免因为卷积运算导致输出图像缩小和图像边缘信息丢失，常常采用图像边缘填充技术，即在图像四周边缘填充0，使得卷积运算后图像大小不会缩小，同时也不会丢失边缘和角落的信息。

array：array_like,要填充的数组

pad_width: {sequence, array_like, int},填充的长度,可以是序列、数组、整数类型

如果是序列，必须是以下形式：

（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第n轴前面填充before_1个、后面填充after_1个数值。

mode: str,填充的方式，共计11种

![image-20230110112455354](README.assets/image-20230110112455354.png)



### numpy.ediff1d(ary, to_end=None, to_begin=None)

返回ary中连续元素之间的差值

to_end、to_begin为array类型，会被加入ary最后、最前，然后在进行计算

### numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

a与b进行element-wise的比较

`absolute(a - b) <= (atol + rtol * absolute(b))`
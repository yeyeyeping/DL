## 正确率、召回率

混淆矩阵

![image-20230719163621312](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230719163621312.png)



在二分类问题中，精度(precision)是指`作出的所有预测`中，预测正确的数量。

![image-20230719220429492](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230719220429492.png)

Precision是预测模型的`主观`性能的度量，也就是说模型预测为正确的样本中，实际为正的占比，`precision的高可以受预测数量的影响`,模型只对相对置信的样本预测，保守的只预测一定对的样本，可以保证高的precision。

Recall也叫`查全率`，是预测模型的`客观`性能的度量，就是实际为正例的样本中真的为正的数量的占比。

`Precision与Recall无论哪一个都不能单独描述模型性能`，两者`同时高`才能保证效果好，因此引入了F1 score，

`F1-score是Precision与Recall的调和平均`,调和平均对于0-1之间的数很敏感，调和平均越大，precision与recall都大

![image-20230719220416822](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230719220416822.png)

![image-20230719220444489](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230719220444489.png)

==补充：调和平均==

![image-20230719222250549](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230719222250549.png)

## F1-score与Dice系数

Dice就是F1-score

<img src="/home/yeep/project/py/DL-master/BasicDL/assets/image-20230720135843912.png" alt="image-20230720135843912" style="zoom: 67%;" />
$$
FSocre=\frac{2}{\frac{TP+FP}{TP}+\frac{TP+FN}{TP}}
=\frac{2TP}{2TP+FP+FN}
$$

$$
Dice = \frac{2*pred*mask+eps}{pred+mask+eps}
$$



而：

pred*mask -> TP

TP+FP ->  sum(pred)

TP+FN -> sum(mask)

nnunet对于Dice Loss的实现：

![image-20230720140224628](/home/yeep/project/py/DL-master/BasicDL/assets/image-20230720140224628.png)

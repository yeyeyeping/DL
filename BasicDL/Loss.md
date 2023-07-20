## Dice Loss

$$
Dice=FSocre=\frac{2TP}{2TP+FP+FN}
$$

## Tversky loss

在分割任务中，`FN出现的概率远远高与FN`，但在Dice损失函数中，FP与FN占有相同的权重，这是不合理的。
$$
Tversky \ loss =1- \frac{TP}{TP+\alpha FP+ \beta FN}
$$
$\alpha = \beta = 0.5$ 时，Tversky loss与Dice损失函数等价。

网络训练的目的时同时降低FP，FN，如果$\beta$更大，则降低FN带来的收益更大，因此模型在FN上的表现更好，更关注recall的优化


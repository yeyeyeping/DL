# Torch.nn

## torch.nn.parameter.Parameter(data=None, requires_grad=True)

Tensor的子类，被作为Module的参数。如果这个类通过赋值运算符出现在Module的属性，会被自动添加到Module的paramters中，可以被parameters()或named_parameters()所迭代。但赋值Tensor类的对象给Module时，没有这个作用。例如，linear层的参数：

```python
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
```



## torch.nn.parameter.UninitializedParameter(requires_grad=True, device=None, dtype=None)

UninitializedParameter是没有被初始化的Parameter，没有数据，访问属性会抛出异常。`只支持改变dtype、device、转换成Parameter三个操作`。常常会出现在Lazyxxx(例如Lazylinear,输出维度自己设定，输入维度会在第一次forward的时候自动初始化)等Module中.

使用materialize方法来初始化

materialize(self, shape: Tuple[int, ...], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None)

```python
class LazyLinear(LazyModuleMixin, Linear):

    cls_to_become = Linear  # type: ignore[assignment]
    weight: UninitializedParameter
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(self, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(0, 0, False)
        self.weight = UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = UninitializedParameter(**factory_kwargs)
        def initialize_parameters(self, input) -> None:  # type: ignore[override]
            if self.has_uninitialized_params():
                with torch.no_grad():
                    self.in_features = input.shape[-1]
                    self.weight.materialize((self.out_features, self.in_features))
                    if self.bias is not None:
                        self.bias.materialize((self.out_features,))
                    self.reset_parameters()
```

## orch.nn.parameter.UninitializedBuffer(requires_grad=False, device=None, dtype=None)


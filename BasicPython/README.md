## Python的可迭代对象(iterable)、迭代器(iterator)、生成器(generator)的区别

1. 实现了`__iter__`方法能返回一个`迭代器(iterator)`的对象都是可迭代对象(iterable)

2. 而迭代器(iterator)是实现了`__iter__`方法与`__next__`方法的对象,在迭代器中，`__iter__`中只需要返回自身，看似毫无意义，实际上，这样的设计是为了`统一容器和迭代器在for in表达式的行为`。for in执行时，无论是容器还是迭代器，都是先执行`对象.__iter__或者iter(对象)`方法，获取可迭代对象，然后不断调用`next()`方法获取下一个元素。
3. 生成器是以一种便捷的方法实现的迭代器的魔术方法(`__iter__`、`__next__`)，是一种特殊的迭代器。其通常结合yield表达式使用，如果某个方法使用了yield表达式，该方法调用时将会被封装成迭代器，因此某个类中的`__iter__`方法使用yield方法返回数值，就不用返回可迭代对象，会被自动实现

## 命令规范

![img](assets/eb824bef6b2c9.jpeg)

## Typing




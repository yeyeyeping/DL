# -*- coding: utf-8 -*-
class My:

    def __init__(self) -> None:
        super().__init__()
        self.count = [1,2,3,4]
    def __iter__(self):
        return iter(self.count)
m = My()
for i in m:
    print(i)
dir()
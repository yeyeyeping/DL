# -*- coding: utf-8 -*-
import time

import torch, os
from torch.nn import Module, Linear
import importlib


def save_checkpoint(checkpoint_dir: str, cur_epoch: int, model: Module, metric: dict,
                    optimizer: Module = None,
                    optimizer_param: dict = None,
                    ) -> None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    pick = {
        'epoch': cur_epoch + 1,
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        pick['optimizer'] = {
            "module": optimizer.__module__,
            "name": optimizer.__class__.__name__,
            "state_dict": optimizer.state_dict(),
            "param": optimizer_param
        }
    subfix = f"checkpoint_{cur_epoch}_timestamp_{time.time()}"
    for k, v in metric.items():
        subfix += f"_{k}{v:.4f}"
    filename = f'{subfix}.pth.tar'
    torch.save(pick, os.path.join(checkpoint_dir, filename))


def recover_from(ckpath: str, model: Module, map_location=None):
    assert os.path.exists(ckpath), "checkpoint not exists"
    ckdict = torch.load(ckpath, map_location=map_location)

    model.load_state_dict(ckdict["state_dict"])
    if "optimizer" not in ckdict:
        return {
            "epoch":ckdict["epoch"]
        }
    optim_package = importlib.import_module(ckdict["optimizer"]["module"])
    optimizer = getattr(optim_package, ckdict["optimizer"]["name"])(model.parameters(),
                                                                    **ckdict["optimizer"]["param"])
    optimizer.load_state_dict(ckdict["optimizer"]["state_dict"])
    return {
        "epoch": ckdict["epoch"],
        "optimizer": optimizer
    }


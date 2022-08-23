import argparse
import os

import megengine as mge
import numpy as np
import torch
import torch.nn as nn
from models.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from models.torch_models import LightCNN_9Layers as torch_LightCNN_9Layers
from models.torch_models import LightCNN_29Layers as torch_LightCNN_29Layers
from models.torch_models import LightCNN_29Layers_v2 as torch_LightCNN_29Layers_v2


MODEL_MAPPER = {
    '9': (LightCNN_9Layers, torch_LightCNN_9Layers),
    '29': (LightCNN_29Layers, torch_LightCNN_29Layers),
    '29v2': (LightCNN_29Layers_v2, torch_LightCNN_29Layers_v2),
}

def get_atttr_by_name(torch_module, k):
    name_list = k.split('.')
    sub_module = getattr(torch_module, name_list[0])
    if len(name_list) != 1:
        for i in name_list[1:-1]:
            try:
                sub_module = getattr(sub_module, i)
            except:
                sub_module = sub_module[int(i)]
    return sub_module


def convert(torch_model, torch_dict):
    new_dict = {}
    for k, v in torch_dict.items():
        data = v.numpy()
        sub_module = get_atttr_by_name(torch_model, k)
        is_conv = isinstance(sub_module, nn.Conv2d)
        if is_conv:
            groups = sub_module.groups
            is_group = groups > 1
        else:
            is_group = False
        if "weight" in k and is_group:
            out_ch, in_ch, h, w = data.shape
            data = data.reshape(groups, out_ch // groups, in_ch, h, w)
        if "bias" in k:
            if is_conv:
                data = data.reshape(1, -1, 1, 1)
        if "num_batches_tracked" in k:
            continue
        new_dict[k] = data
    return new_dict


def main(torch_name, torch_path):
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    torch_state_dict = torch_state_dict['state_dict']
    s = {}
    for k in torch_state_dict.keys():
        s[k.replace("module.", "")] = torch_state_dict[k]
    torch_model = MODEL_MAPPER[torch_name][1]()
    torch_model.load_state_dict(s)
    model = MODEL_MAPPER[torch_name][0]()

    new_dict = convert(torch_model, s)
    model.load_state_dict(new_dict)
    os.makedirs('pretrained', exist_ok=True)
    mge.save(new_dict, os.path.join('pretrained', torch_name + '.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='29',
        help=f"which model to convert from torch to megengine, optional: {list(MODEL_MAPPER.keys())}",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        default="./LightCNN_29Layers_checkpoint.pth.tar",
        help=f"path to torch checkpoint",
    )
    args = parser.parse_args()
    main(args.model, args.ckpt)

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np
import torch
import torch.nn as nn
from models.pooling import MaxPool2d
from models.light_cnn import mfm
from models.torch_models import mfm as tmfm
mge.config.async_level = 0

def test_func(mge_out, torch_out):
    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err

inp = np.random.randn(2, 3, 224, 224)
t_inp = torch.from_numpy(inp).float()
m_inp = mge.tensor(inp, dtype=np.float32)

m = mfm(3, 64, 3, 1, 1)
t = tmfm(3, 64, 3, 1, 1)

mge_out = m(m_inp)
torch_out = t(t_inp)
ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out)
print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

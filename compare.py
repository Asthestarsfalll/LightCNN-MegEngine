import time

import megengine as mge
import numpy as np
import torch

from convert_weights import convert, MODEL_MAPPER
model_name = '9'

mge_model = MODEL_MAPPER[model_name][0](True)
torch_model = MODEL_MAPPER[model_name][1]()

# mge_model.load_state_dict(mge.load('./pretrained/9.pkl'))
# t = torch.load(
#     "./LightCNN_9Layers_checkpoint.pth.tar", map_location='cpu')['state_dict']
# s = {}
# for k in t.keys():
#     s[k.replace("module.", "")] = t[k]
# torch_model.load_state_dict(s)

torch_state_dict = torch_model.state_dict()
new_dit = convert(torch_model, torch_state_dict)
mge_model.load_state_dict(new_dit)

mge_model.eval()
torch_model.eval()


torch_time = meg_time = 0.0


def test_func(mge_out, torch_out, func=None):
    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    if func is not None:
        mge_out = func(mge_out)
        torch_out = func(torch_out)
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def softmax(logits):
    logits = logits - logits.max(-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(-1, keepdims=True)


if torch.cuda.is_available():
    torch_inp = torch_inp.cuda()
    torch_model.cuda()


for i in range(15):
    results = []
    inp = np.random.randn(2, 1, 128, 128)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)


    st = time.time()
    mge_out = mge_model(mge_inp)[0]
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)[0]
    torch_time += time.time() - st

    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out, func=softmax)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

assert all(results), "not aligned"

print(f"meg time: {meg_time}, torch time: {torch_time}")

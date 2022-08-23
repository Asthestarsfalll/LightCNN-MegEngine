import megengine as mge
import megengine.functional as F
import megengine.module as M


class MaxPool2d(M.MaxPool2d):
    """only used for this network, not universal"""
    def __init__(self, ceil_mode=False, **kwargs):
        super(MaxPool2d, self).__init__(**kwargs)
        self.ceil_mode = ceil_mode
    
    def forward(self, inp):
        if self.ceil_mode:
            if inp.shape[2] %  2 != 0:
                inp = F.nn.pad(inp, ((0, 0), (0, 0), (0, 1), (0, 1)), constant_value=-float('inf'))
                print(inp.shape)
        return super().forward(inp)
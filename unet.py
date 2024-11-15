from torch.hub import load
from torch.nn import Softmax





def unet_model():
    model=load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.add_module("softmax",Softmax())
    return model

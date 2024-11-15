from torch.hub import load
from torch.nn import Softmax,Conv2d





def unet_model():
    model=load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.conv=Conv2d(32,4,kernel_size=(1,1),stride=(1,1))
    model.add_module("softmax",Softmax())
    for name,param in model.named_parameters():
            param.requires_grad=True
    for param in model.conv.parameters():
        param.requires_grad=True
    return model



model=unet_model()
import torch
from torch.utils.data import Dataset,DataLoader
import os
from torchvision.transforms.v2 import Compose,ToTensor,Normalize
from PIL import Image
import numpy as np
import matplotlib.image as Imagemat 
from viewer import torch_to_image
sample="Banepa Tunnel Feasibility Survey_transparent_mosaic_group127757.PNG"

PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/Test_images/"

def sample_output_generator(model):
    model.to('cpu')
    PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/Test_images/"
    sample="Banepa Tunnel Feasibility Survey_transparent_mosaic_group128191.PNG"
    image=Image.open(PATH+sample)
    image=image.resize((256,256))
    transformer_image=Compose([
            ToTensor(),
            #Normalize(mean=mean, std=std),
            ])
    image=image.convert("RGB")
    image=transformer_image(image)
    with torch.no_grad():
        pred=model(image.unsqueeze(0))
        print(pred.shape)
        torch_to_image(image,pred.squeeze(0),"sample_prediction_new.png")
        
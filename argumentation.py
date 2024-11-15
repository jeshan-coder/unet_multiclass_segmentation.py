import torchvision.transforms.v2 as transforms
import os
from progressbar import ProgressBar 
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import random
TRAIN_MASK="/home/jeshanpokharel/Downloads/WhatSie/Fiveimageswithmask/NEW_MASK/"
TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/WhatSie/Fiveimageswithmask/NEW_IMAGE/"


IMAGE_ARG_PATH="/home/jeshanpokharel/Downloads/WhatSie/Fiveimageswithmask/IMAGE_ARG_PATH/"
MASK_ARG_PATH="/home/jeshanpokharel/Downloads/WhatSie/Fiveimageswithmask/MASK_ARG_PATH/"



train_images_=sorted(os.listdir(TRAIN_IMAGES_PATH))
mask_images_=sorted(os.listdir(TRAIN_MASK))
i=0
aug_bar=ProgressBar(max_value=len(train_images_),prefix="Argumentation")
for image,mask in zip(train_images_,mask_images_):
    flip=random.choice([0.1,0.2,0.3,0.4,0.5,.6,0.7,0.8])
    contrast=random.choice([0.1,0.2,0.3,0.4,0.5,0.6])
    image_=Image.open(TRAIN_IMAGES_PATH+image)
    mask_=Image.open(TRAIN_MASK+mask)
    image_=np.array(image_)
    mask_=np.array(mask_)
    
    for j in range(5):
        transform_arg = transform = A.Compose([
            A.HorizontalFlip(p=flip),
            A.RandomBrightnessContrast(p=contrast),
            ])
        transformed=transform_arg(image=image_,mask=mask_)
        image_arg,mask_arg=transformed['image'],transformed['mask']
        image_arg,mask_arg=Image.fromarray(image_arg),Image.fromarray(mask_arg)
        image_arg.save(IMAGE_ARG_PATH+f"{i}.png")
        mask_arg.save(MASK_ARG_PATH+f"{i}.png")
        i+=j
    aug_bar.next()
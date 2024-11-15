"""mask generation"""
import os
import numpy as np
from PIL import Image
from progressbar import ProgressBar 
IMAGE_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/Training_images/"
MASK_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/Training_mask/"


NEW_IMAGE_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/NEW_IMAGE/"
NEW_MASK_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/MASK_IMAGE/"



list_of_images=sorted(os.listdir(IMAGE_PATH))

list_of_mask=sorted(os.listdir(MASK_PATH))

print(list_of_mask)
aug_bar=ProgressBar(max_value=len(list_of_images),prefix="NEW IMAGES")
i=0
for image,mask in zip(list_of_images,list_of_mask):
    #load image and mask
    img_=Image.open(IMAGE_PATH+image)
    mask_=Image.open(MASK_PATH+mask)
    
    img_arr=np.array(mask_)
    shape_of_img=img_arr.shape
    img_arr=img_arr.flatten()


    img_arr[img_arr==255]=1
    img_arr[img_arr==85]=2
    img_arr[img_arr==170]=3

    img_arr=img_arr.reshape(shape_of_img[0],shape_of_img[1])
    img=Image.fromarray(img_arr)
    img.save(NEW_MASK_PATH+f"{mask}.png")
    img_.save(NEW_IMAGE_PATH+f"{image}.png")
    aug_bar.next()
    i+=1



from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Binarizer
from viewer import view_image,torch_to_image
import numpy as np
import matplotlib.image as Imagemat 
from datapreparation import UnetDataPreparation
from PIL import Image
"""IMAGE AND MASK PATH"""

#MASK="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/label_images_semantic/"
#IMAGES="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/original_images/"


SAMPLE_MASK="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/label_images_semantic/041.png"
SAMPLE_IMAGE="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/original_images/041img_arr
.png"




view_image(SAMPLE_IMAGE,SAMPLE_MASK,1)

# get unique values of images
img=Image.open(SAMPLE_MASK)
img_arr=np.array(img)
shape_of_img=img_arr.shape
img_arr=img_arr.reshape(-1,1)

#one hot encoder
#onehot=OneHotEncoder(sparse_output=False)
#one_hot=onehot.fit(img_arr)


#img_onehot=one_hot.transform(img_arr)
#img_onehot=img_onehot.reshape(shape_of_img[0],shape_of_img[1],-1)


#dataset=UnetDataPreparation(IMAGES,MASK)


#sample_image,sample_mask=next(iter(dataset))

#torch_to_image(sample_image,sample_mask,1)







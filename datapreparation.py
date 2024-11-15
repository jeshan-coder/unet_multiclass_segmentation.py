from torch.utils.data import Dataset,DataLoader
import os
from torchvision.transforms.v2 import Compose,ToTensor,Normalize
from PIL import Image
import numpy as np
import matplotlib.image as Imagemat 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Binarizer
class UnetDataPreparation(Dataset):
    def __init__(self,image_path,mask_path):
        super().__init__()
            
        self.image_path=image_path
        self.mask_path=mask_path
            
            
        #sort images
            
        self.images=os.listdir(self.image_path)
        self.images=sorted(self.images)
            
            
        #sort masks
            
        self.masks=os.listdir(self.mask_path)
        self.masks=sorted(self.masks)
            
        
        #one hot initialization
        self.onehot=OneHotEncoder(sparse_output=False,categories=[[0,1,2,3]])
            
        #transformation
            
    
    
    def __len__(self):
        
        return len(self.masks)
    
    
    def mask_one_hot_encode(self,mask_pil_object):
        #image array
        img_arr=np.array(mask_pil_object)
        
        
        #shape of image
        shape_of_img=img_arr.shape
        
        
        #reshape 
        img_arr=img_arr.reshape(-1,1)
        
        #onehot
        one_hot=self.onehot.fit(img_arr)
        
        #transform
        img_onehot=one_hot.transform(img_arr)
        
        
        #reshape to H,W,C(i.e categories)
        img_onehot=img_onehot.reshape(shape_of_img[0],shape_of_img[1],-1)
        
        
        return img_onehot
        
        
    
    def __getitem__(self,index):
        
        
        # get image name
        image=self.images[index]
        mask=self.masks[index]
        
        #resize images
        
        # open imaage and mask
        image=Image.open(self.image_path+image)
        mask=Image.open(self.mask_path+mask)
        mask=mask.convert('L')
        
        #resize image and mask
        image=image.resize((256,256))
        mask=mask.resize((256,256))
        
        mask=self.mask_one_hot_encode(mask)
        
        
        #calculate mean and std
        mean, std = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))
        
        #transformer initialization
        
        #image
        transformer_image=Compose([
                ToTensor(),
                #Normalize(mean=mean, std=std),
                ])
        
        #mask
        transformer_mask=Compose([
                ToTensor(),
                ])
        
        #convert image to rgb only
        image=image.convert("RGB")
        
        #convert mask to gray value only (not necessary)
        #mask=mask.convert('L')
        
        #convert image and mask to torch
        image=transformer_image(image)
        mask=transformer_mask(mask)
        
        return image,mask

        
        
        
        
        
        
        
        
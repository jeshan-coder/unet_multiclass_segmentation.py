"""unet main segnemtation"""
from torch.utils.data import DataLoader
from datapreparation import UnetDataPreparation
from my_unet import unet_model
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss
from torch.optim import Adam
from viewer import compare_mask,torch_to_image
from progressbar import ProgressBar
import progressbar
from my_accuracy_measurement import iou_score_per_patch
import warnings
import torch
from test_acuracy import accuracy_test
from torch.optim.lr_scheduler import StepLR
from sample_output_generator import sample_output_generator
warnings.filterwarnings('ignore')
"""FILE PATHS"""
#TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/carvana-image-masking-challenge/train/"
#TRAIN_MASK="/home/jeshanpokharel/Downloads/carvana-image-masking-challenge/train_masks/"

#TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/khimsir new data/train_images/"
#TRAIN_MASK="/home/jeshanpokharel/Downloads/khimsir new data/train_masks/"


#TRAIN_MASK="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/label_images_semantic/"
#TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/original_images/"
device="cuda"


#IMAGE_ARG_PATH="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/ARGUMENTATION_IMAGES/"
#MASK_ARG_PATH="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/ARGUMENTATION_MASKS/"

#IMAGE_ARG_PATH="/home/jeshanpokharel/Downloads/WhatSie/Fiveimageswithmask/IMAGE_ARG_PATH/"
#MASK_ARG_PATH="/home/jeshanpokharel/Downloads/WhatSie/Fiveimageswithmask/MASK_ARG_PATH/"

IMAGE_ARG_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/NEW_IMAGE/"
MASK_ARG_PATH="/home/jeshanpokharel/Downloads/WhatSie/test_training_images_along_withmasks_setof_upto_900_images_oct_28/Khempratikpractice/MASK_IMAGE/"
# train datasets and dataloader
train_dataset=UnetDataPreparation(IMAGE_ARG_PATH,MASK_ARG_PATH)
#train_dataset=UnetDataPreparation(TRAIN_IMAGES_PATH,TRAIN_MASK)
train_dataloader=DataLoader(train_dataset,batch_size=8)



# test datasets and dataloader
test_dataset=UnetDataPreparation(IMAGE_ARG_PATH,MASK_ARG_PATH)
test_dataloader=DataLoader(test_dataset,batch_size=8)


#define model
model=unet_model()
for name,param in model.named_parameters():
        print(name,param.requires_grad)
model=model.to(device)
#load loss and optimizer
loss_fn=CrossEntropyLoss()
optimizer=Adam(model.parameters())


scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
#widget for progress bar
widgets = [
    ' [', progressbar.Percentage(), '] ',
    ' (', progressbar.ETA(), ') ',
    ' Loss: ', progressbar.Variable('loss'),  # Display loss
    ' Accuracy: ', progressbar.Variable('accuracy'),  # Display accuracy
]





#train model
n_epochs=30
loss_best=100
threshold=100
i=0
for epoch in range(n_epochs):
    progress=ProgressBar(min_value=0,max_value=len(train_dataloader),widgets=widgets,prefix="Training")
    for train_x,train_y in train_dataloader:
        train_x,train_y=train_x.to(device),train_y.to(device)
        pred=model(train_x)
        #visualizer
        compare_mask(train_y, pred,epoch)
        
        #accuracy calculaltion per batch
        with torch.no_grad():
            acc=iou_score_per_patch(train_y,pred)
        
        
        loss=loss_fn(pred,train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #update progress and variables
        progress.update(loss=loss.item(),accuracy=acc)
        progress.next()
    scheduler.step()
    progress.finish()
    #test_acc=accuracy_test(test_dataloader, loss_fn,model)
    if i<=threshold:
        if loss.item()<loss_best:
            loss_best=loss.item()
            torch.save(model.state_dict(),"model.pth")
            print("Saved !")
        elif loss.item()>loss_best:
            i+=1
    print(f"{epoch} loss:{round(loss.item(),4)} accuracy:{round(acc,4)} test accuracy:{round(0,4)}")
    print("\n")
sample_output_generator(model)
    

    
        





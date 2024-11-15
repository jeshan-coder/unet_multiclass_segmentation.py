import torch
from my_accuracy_measurement import iou_score_per_patch
import numpy as np
from progressbar import ProgressBar
import progressbar
from viewer import torch_to_image



TRAIN_MASK="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/label_images_semantic/"
TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/archive/classes_dataset/classes_dataset/original_images/"
device="cuda"






#widget for progress bar
widgets = [
    ' [', progressbar.Percentage(), '] ',
    ' (', progressbar.ETA(), ') ',
    ' Loss: ', progressbar.Variable('loss'),  # Display loss
    'testAccuracy: ', progressbar.Variable('accuracy'),  # Display accuracy
]

def accuracy_test(test_dataloader,loss_fn,model):
    epoch_acc=[]
    progress=ProgressBar(min_value=0,max_value=len(test_dataloader),widgets=widgets,prefix="Validation")
    for train_x,train_y in test_dataloader:
        train_x,train_y=train_x.to(device),train_y.to(device)
        
        with torch.no_grad(): 
            pred=model(train_x)
            loss=loss_fn(pred,train_y)
            acc=iou_score_per_patch(train_y,pred)
            epoch_acc.append(acc)
        progress.update(loss=loss.item(),accuracy=acc)
        progress.next()
    epoch_acc=np.array(epoch_acc)
    return float(np.mean(epoch_acc))
    

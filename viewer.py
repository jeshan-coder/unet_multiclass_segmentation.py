import matplotlib.pyplot as plt
import matplotlib.image as Image 


SAVE_PATH="/home/jeshanpokharel/notebooks/unet_multiclass_segmentation.py/Subplots/"

def view_image(path_image,path_mask,index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    image=Image.imread(path_image)
    mask_image=Image.imread(path_mask)
    #image shower subplot
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis('off')
    #mask shower
    ax2.imshow(mask_image)
    ax2.set_title("Mask")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_PATH+f"{index}.png")

    

def torch_to_image(tensor_image,tensor_mask,index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    
    #tensor image
    image=tensor_image.permute(1,2,0)
    
    #mask image
    mask=tensor_mask.argmax(dim=0)
    
    
    #read using plt image
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis('off')
    
    #mask shower
    ax2.imshow(mask)
    ax2.set_title("Mask")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_PATH+f"{index}.png",dpi=300)
    


def compare_mask(true_mask,pred_mask,index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    
    #tensor image
    true_mask=true_mask[0].detach().cpu()
    pred_mask=pred_mask[0].detach().cpu()
    
    #true mask image
    true_mask=true_mask.argmax(dim=0)
    
    #pred mask image
    pred_mask=pred_mask.argmax(dim=0)
    
    #read using plt image
    ax1.imshow(true_mask)
    ax1.set_title("TRUE IMAGE")
    ax1.axis('off')
    
    #mask shower
    ax2.imshow(pred_mask)
    ax2.set_title("PRED IMAGE")
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(SAVE_PATH+f"{index}.png",dpi=300)
    
    
    










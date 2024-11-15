from sklearn.metrics import confusion_matrix
import numpy as np

def iou_score_per_patch(y_true, y_pred, num_classes=4):
    # Convert tensors to numpy arrays
    y_true = y_true.detach().cpu().numpy()  # Shape: [batch_size, num_classes, height, width]
    y_pred = y_pred.detach().cpu().numpy()

    batch_size = y_true.shape[0]  # Number of images in the batch
    iou_per_patch = []

    for i in range(batch_size):
        # Flatten the true and predicted masks for each image
        y_true_flat = np.argmax(y_true[i], axis=0).flatten()  # Flatten the ground truth
        y_pred_flat = np.argmax(y_pred[i], axis=0).flatten()  # Flatten the predicted mask
        
        # Compute confusion matrix for each image (patch)
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(num_classes))
        
        # Initialize IoU array to store per-class IoU
        iou = np.zeros(num_classes)
        
        for c in range(num_classes):
            intersection = cm[c, c]  # True positives for class c
            union = cm[c, :].sum() + cm[:, c].sum() - intersection  # Union for class c
            iou[c] = intersection / float(union) if union != 0 else 0  # Avoid division by zero
        
        # Add IoU for this image to the list
        iou_per_patch.append(iou)
    
    # Convert list to numpy array for easier manipulation
    iou_per_patch = np.array(iou_per_patch)
    
    # Mean IoU across all images in the batch, and average over all classes
    mean_iou_batch = np.mean(iou_per_patch, axis=0)
    mean_iou = np.mean(mean_iou_batch)  # Mean IoU for the whole batch
    
    return mean_iou

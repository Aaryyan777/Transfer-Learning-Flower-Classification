import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper 'Focal Loss for Dense Object Detection'.
    
    Args:
        gamma (float): Focusing parameter. Higher values give more weight to hard-to-classify examples.
        alpha (float): Balancing parameter. Balances the importance of positive/negative examples.
    
    Returns:
        A callable loss function.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Ensure predictions are clipped to avoid log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        
        # Sum the loss over the classes
        return K.sum(loss, axis=-1)
        
    return focal_loss_fixed

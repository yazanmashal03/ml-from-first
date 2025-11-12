import numpy as np

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

def accuracy(y_true, y_pred):
    """Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def precision(y_true, y_pred, positive_label=1):
    """Calculate precision: TP / (TP + FP)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true_binary = (y_true == positive_label).astype(int)
    y_pred_binary = (y_pred == positive_label).astype(int)
    
    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    
    # Avoid division by zero
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred, positive_label=1):
    """Calculate recall: TP / (TP + FN)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Convert to binary: positive_label -> 1, others -> 0
    y_true_binary = (y_true == positive_label).astype(int)
    y_pred_binary = (y_pred == positive_label).astype(int)
    
    # True Positives: predicted positive and actually positive
    tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    # False Negatives: predicted negative but actually positive
    fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
    
    # Avoid division by zero
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def compute_reconstruction_error(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) reconstruction error between original and predicted data.
    Input shapes: (num_samples, window_size, num_features)
    Output shape: (num_samples,)
    """
    errors = np.mean(np.square(y_true - y_pred), axis=(1, 2))
    return errors

def calculate_anomaly_threshold(errors, k=3):
    """
    Calculate the anomaly threshold using mean + k * std_dev.
    """
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + k * std_error
    return threshold

def evaluate_metrics(y_true, y_pred_labels):
    """
    Evaluate precision, recall, f1-score and return a confusion matrix.
    y_true: True binary labels (0 Normal, 1 Anomaly)
    y_pred_labels: Predicted binary labels (0 Normal, 1 Anomaly)
    """
    precision = precision_score(y_true, y_pred_labels, zero_division=0)
    recall = recall_score(y_true, y_pred_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_labels)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return metrics

import torch
from sklearn.metrics import accuracy_score, mean_squared_error as mse, mean_absolute_error as mae

# --- Core Metric Functions ---

def accuracy(outputs, labels):
    """Calculates accuracy for classification tasks."""
    # Handle outputs from different loss functions (e.g., CrossEntropyLoss vs BCEWithLogitsLoss)
    if outputs.ndim == 2 and outputs.shape[1] > 1: # Multi-class (logits or probabilities)
        preds = torch.argmax(outputs, dim=1)
    elif outputs.ndim == 1 or outputs.shape[1] == 1: # Binary (logits or probabilities)
        preds = (outputs > 0.5).float() # Assuming sigmoid output -> threshold at 0.5
        labels = labels.float() # Ensure labels are float for comparison
    else:
        raise ValueError(f"Unsupported output shape for accuracy calculation: {outputs.shape}")

    # Ensure labels and preds are on CPU and numpy for sklearn compatibility if needed
    # Or implement directly with torch
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

def mean_squared_error(outputs, targets):
    """Calculates Mean Squared Error for regression tasks."""
    # Ensure tensors are float and have compatible shapes
    outputs = outputs.float().squeeze()
    targets = targets.float().squeeze()
    if outputs.shape != targets.shape:
         # Attempt to fix common shape mismatches (e.g., [N, 1] vs [N])
         if outputs.numel() == targets.numel():
             outputs = outputs.view_as(targets)
         else:
             raise ValueError(f"Shape mismatch for MSE: outputs {outputs.shape}, targets {targets.shape}")
    # return mse(targets.cpu().numpy(), outputs.cpu().numpy()) # Using sklearn
    return torch.mean((outputs - targets) ** 2).item() # Using torch

def mean_absolute_error(outputs, targets):
    """Calculates Mean Absolute Error for regression tasks."""
     # Ensure tensors are float and have compatible shapes
    outputs = outputs.float().squeeze()
    targets = targets.float().squeeze()
    if outputs.shape != targets.shape:
         # Attempt to fix common shape mismatches
         if outputs.numel() == targets.numel():
             outputs = outputs.view_as(targets)
         else:
            raise ValueError(f"Shape mismatch for MAE: outputs {outputs.shape}, targets {targets.shape}")

    # return mae(targets.cpu().numpy(), outputs.cpu().numpy()) # Using sklearn
    return torch.mean(torch.abs(outputs - targets)).item() # Using torch

# --- Metric Calculation Orchestrator ---

def calculate_metrics(outputs, targets, task_type):
    """
    Calculates relevant metrics based on the task type.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth labels/values.
        task_type (str): 'classification' or 'regression'.

    Returns:
        dict: Dictionary containing calculated metric names and values.
    """
    metrics = {}
    if task_type == 'classification':
        # Ensure outputs and targets are suitable for accuracy calculation
        metrics['accuracy'] = accuracy(outputs, targets)
        # Add other classification metrics if needed (e.g., precision, recall, F1)
        # from sklearn.metrics import precision_score, recall_score, f1_score
        # try:
        #     labels_np = targets.cpu().numpy()
        #     preds_np = torch.argmax(outputs, dim=1).cpu().numpy() if outputs.ndim == 2 and outputs.shape[1] > 1 else (outputs > 0.5).float().cpu().numpy()
        #     metrics['precision'] = precision_score(labels_np, preds_np, average='weighted', zero_division=0)
        #     metrics['recall'] = recall_score(labels_np, preds_np, average='weighted', zero_division=0)
        #     metrics['f1'] = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        # except Exception as e:
        #     print(f"Warning: Could not calculate precision/recall/F1: {e}")

    elif task_type == 'regression':
        metrics['mse'] = mean_squared_error(outputs, targets)
        metrics['mae'] = mean_absolute_error(outputs, targets)
        # Add other regression metrics if needed (e.g., R-squared)
        # from sklearn.metrics import r2_score
        # try:
        #     targets_np = targets.cpu().numpy()
        #     outputs_np = outputs.cpu().numpy()
        #     metrics['r2'] = r2_score(targets_np, outputs_np)
        # except Exception as e:
        #      print(f"Warning: Could not calculate R2 score: {e}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return metrics
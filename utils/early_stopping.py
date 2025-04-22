import numpy as np
import torch # Only needed if saving model directly here, but we use CheckpointManager

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, config, verbose=True):
        """
        Args:
            config (dict): Configuration dictionary containing early stopping parameters.
                           Expected keys: 'enabled', 'patience', 'mode', 'metric', 'delta'.
            verbose (bool): If True, prints a message for each validation metric improvement.
        """
        es_config = config.get('early_stopping', {})
        self.enabled = es_config.get('enabled', False)
        if not self.enabled:
            print("Early stopping disabled.")
            return # Don't initialize further if disabled

        self.patience = es_config.get('patience', 5)
        self.mode = es_config.get('mode', 'min').lower()
        self.metric_key = es_config.get('metric', 'val_loss') # Metric name in logs to monitor
        self.delta = es_config.get('delta', 0.0)
        self.verbose = verbose

        if self.mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")

        self.counter = 0
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.early_stop = False

        print(f"Early stopping enabled: patience={self.patience}, mode='{self.mode}', metric='{self.metric_key}', delta={self.delta}")

    def __call__(self, current_metric_value):
        """
        Checks if early stopping criteria are met.

        Args:
            current_metric_value (float): The validation metric score for the current epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if not self.enabled:
            return False

        score = current_metric_value

        improvement = False
        if self.mode == 'min':
            if score < self.best_score - self.delta:
                improvement = True
        else: # mode == 'max'
            if score > self.best_score + self.delta:
                improvement = True

        if improvement:
            # self.best_score = score # Best score is now managed by Trainer/CheckpointManager
            self.counter = 0
            # No need to print improvement here, Trainer does it
            # if self.verbose:
            #     print(f'EarlyStopping: Validation metric improved ({self.best_score:.6f} --> {score:.6f}). Resetting counter.')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_score:.4f}, Current: {score:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def is_improvement(self, current_metric_value, best_metric_so_far):
        """Checks if the current metric value is an improvement over the best seen so far."""
        if self.mode == 'min':
            return current_metric_value < best_metric_so_far - self.delta
        else: # mode == 'max'
            return current_metric_value > best_metric_so_far + self.delta

    def reset(self):
        """Resets the counter and best score."""
        self.counter = 0
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.early_stop = False
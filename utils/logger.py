import os
import csv
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd # Using pandas for easier CSV handling

class Logger:
    """Handles logging to TensorBoard and CSV."""
    def __init__(self, log_dir, run_name, config):
        """
        Args:
            log_dir (str): Base directory for logs.
            run_name (str): Specific name for this run (used for subfolder).
            config (dict): The experiment configuration to log.
        """
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_log_dir = log_dir
        self.log_path = os.path.join(log_dir, self.run_name)

        # Create directories
        self.tensorboard_dir = os.path.join(self.log_path, 'tensorboard')
        self.csv_dir = os.path.join(self.log_path, 'csv')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        print(f"📊 TensorBoard logs will be saved to: {self.tensorboard_dir}")

        # Initialize CSV logging
        self.csv_path = os.path.join(self.csv_dir, 'log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        # Use a placeholder for fieldnames; will update on first log_metrics call
        self.csv_writer = None
        self.csv_fieldnames = []
        print(f"📄 CSV logs will be saved to: {self.csv_path}")

        # Log configuration
        self.log_config(config)


    def log_config(self, config):
        """Logs the configuration."""
        # Log config to a JSON file
        config_path = os.path.join(self.log_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"⚙️ Configuration saved to: {config_path}")

        # Log hyperparameters to TensorBoard
        # Handle nested dicts - simple approach: flatten or log as text
        config_flat = pd.json_normalize(config, sep='_').to_dict(orient='records')[0]
        try:
            self.writer.add_hparams(config_flat, {}) # Add empty metric dict initially
        except Exception as e:
             print(f"Warning: Could not log hyperparameters to TensorBoard: {e}. Config: {config_flat}")
             # Fallback: log as text
             self.writer.add_text("config", json.dumps(config, indent=4))

    def log_metrics(self, step, metrics, phase='train'):
        """
        Logs metrics to TensorBoard and CSV.

        Args:
            step (int): The current step (e.g., epoch or batch number).
            metrics (dict): Dictionary of metric names to values.
            phase (str): Phase identifier (e.g., 'train', 'val', 'test', 'epoch').
        """
        # --- TensorBoard Logging ---
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Tag format: phase/metric_name (e.g., val/loss, epoch/train_accuracy)
                self.writer.add_scalar(f"{phase}/{key}", value, step)
            # Add handling for other types like images or histograms if needed

        # --- CSV Logging ---
        # Include step and phase in the logged data
        log_data = {'step': step, 'phase': phase, **metrics}

        # Initialize CSV writer on first call or if fields change
        current_fieldnames = list(log_data.keys())
        if self.csv_writer is None or set(current_fieldnames) != set(self.csv_fieldnames):
            self.csv_fieldnames = current_fieldnames
            # Close and reopen if fieldnames change (shouldn't happen often with this structure)
            if self.csv_writer is not None:
                 self.csv_file.close()
                 self.csv_file = open(self.csv_path, 'a', newline='') # Append if reopening

            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
            # Write header only if the file was just created (or reopened with new fields)
            if self.csv_file.tell() == 0:
                self.csv_writer.writeheader()

        try:
             self.csv_writer.writerow(log_data)
             self.csv_file.flush() # Ensure data is written to disk
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            print(f"Data causing error: {log_data}")
            print(f"Expected fieldnames: {self.csv_fieldnames}")


    def close(self):
        """Closes the log writers."""
        if self.writer:
            self.writer.close()
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
        print("Log writers closed.")
import os
import torch
import shutil # For copying best checkpoint

class CheckpointManager:
    """Handles saving and loading model checkpoints."""
    def __init__(self, checkpoint_dir, run_name, config):
        """
        Args:
            checkpoint_dir (str): Base directory for checkpoints.
            run_name (str): Specific name for this run (used for subfolder).
            config (dict): Training configuration.
        """
        self.run_name = run_name or "latest_run"
        self.base_checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, self.run_name)
        self.save_best_only = config.get('save_best_only', True)

        os.makedirs(self.checkpoint_path, exist_ok=True)
        print(f"💾 Checkpoints will be saved to: {self.checkpoint_path}")

        self.best_checkpoint_name = 'model_best.pth.tar'
        self.latest_checkpoint_name = 'checkpoint_latest.pth.tar'

    def save_checkpoint(self, state, is_best):
        """
        Saves the current model state.

        Args:
            state (dict): Contains model state_dict, optimizer state_dict, epoch, etc.
            is_best (bool): If True, saves this as the best model so far.
        """
        latest_filepath = os.path.join(self.checkpoint_path, self.latest_checkpoint_name)
        best_filepath = os.path.join(self.checkpoint_path, self.best_checkpoint_name)

        # Always save the latest checkpoint unless save_best_only is strictly True AND it's not the best
        if not (self.save_best_only and not is_best):
             torch.save(state, latest_filepath)
             # print(f"Checkpoint saved to '{latest_filepath}' (Epoch {state['epoch']})")

        if is_best:
            # Copy the latest checkpoint file to the best checkpoint file
            shutil.copyfile(latest_filepath, best_filepath)
            print(f"🏆 Best checkpoint saved to '{best_filepath}' (Epoch {state['epoch']}, Metric: {state.get('best_metric', 'N/A'):.4f})")


    def load_checkpoint(self, filepath):
        """
        Loads a checkpoint from a file.

        Args:
            filepath (str): Path to the checkpoint file.

        Returns:
            dict: The loaded state dictionary, or None if file not found.
        """
        if os.path.isfile(filepath):
            print(f"Loading checkpoint '{filepath}'")
            try:
                checkpoint = torch.load(filepath, map_location=torch.device('cpu')) # Load to CPU first
                print(f"Checkpoint loaded successfully (Epoch {checkpoint.get('epoch', 'N/A')})")
                return checkpoint
            except Exception as e:
                print(f"Error loading checkpoint from {filepath}: {e}")
                return None
        else:
            print(f"Checkpoint file not found: {filepath}")
            return None

    def load_best_checkpoint(self):
        """Loads the best saved checkpoint."""
        best_filepath = os.path.join(self.checkpoint_path, self.best_checkpoint_name)
        return self.load_checkpoint(best_filepath)

    def load_latest_checkpoint(self):
        """Loads the most recent checkpoint."""
        latest_filepath = os.path.join(self.checkpoint_path, self.latest_checkpoint_name)
        return self.load_checkpoint(latest_filepath)

    def get_best_checkpoint_path(self):
        """Returns the path to the best checkpoint file if it exists."""
        path = os.path.join(self.checkpoint_path, self.best_checkpoint_name)
        return path if os.path.isfile(path) else None

    def get_latest_checkpoint_path(self):
        """Returns the path to the latest checkpoint file if it exists."""
        path = os.path.join(self.checkpoint_path, self.latest_checkpoint_name)
        return path if os.path.isfile(path) else None
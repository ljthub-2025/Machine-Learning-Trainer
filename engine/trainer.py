import torch
import time
import numpy as np
from tqdm import tqdm # Optional progress bar

from utils.metrics import calculate_metrics

class Trainer:
    """
    Handles the training and validation loops.
    """
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, config, logger, early_stopper, checkpoint_manager):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.early_stopper = early_stopper
        self.checkpoint_manager = checkpoint_manager

        self.epochs = config['epochs']
        self.log_interval = config['log_interval']
        self.task_type = config['task'].lower()
        self.start_epoch = 1
        self.best_metric = np.inf if self.early_stopper.mode == 'min' else -np.inf

        # Load checkpoint if exists to resume training (optional)
        # ckpt_path = self.checkpoint_manager.get_latest_checkpoint_path()
        # if ckpt_path:
        #     print(f"Resuming training from {ckpt_path}")
        #     state = self.checkpoint_manager.load_checkpoint(ckpt_path)
        #     self.model.load_state_dict(state['model_state_dict'])
        #     self.optimizer.load_state_dict(state['optimizer_state_dict'])
        #     self.start_epoch = state['epoch'] + 1
        #     self.best_metric = state['best_metric']
        #     self.early_stopper.best_score = state['best_metric']
        #     self.early_stopper.counter = state.get('early_stopping_counter', 0) # Load counter if saved
        #     print(f"Resumed from epoch {self.start_epoch}, best metric: {self.best_metric:.4f}")


    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs on {self.device}...")
        total_start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.time()

            # --- Training Epoch ---
            train_loss, train_metrics = self._train_epoch(epoch)

            # --- Validation Epoch ---
            val_loss, val_metrics = self._validate_epoch(epoch)

            epoch_duration = time.time() - epoch_start_time

            # --- Logging ---
            log_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()}, # Add prefix
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}, # Add prefix
                "epoch_duration_secs": epoch_duration
            }
            self.logger.log_metrics(step=epoch, metrics=log_metrics, phase='epoch')
            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Metrics: {val_metrics} | Time: {epoch_duration:.2f}s")

            # --- Checkpointing ---
            current_metric = val_metrics.get(self.early_stopper.metric_key, val_loss) # Use monitored metric or val_loss
            is_best = self.early_stopper.is_improvement(current_metric, self.best_metric)

            if is_best:
                self.best_metric = current_metric
                print(f"📈 New best validation metric ({self.early_stopper.metric_key}): {self.best_metric:.4f}")

            # Save checkpoint (best or latest)
            self.checkpoint_manager.save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': self.best_metric,
                    'config': self.config,
                    # 'early_stopping_counter': self.early_stopper.counter # Optional: save ES counter
                },
                is_best=is_best
            )

            # --- Early Stopping ---
            if self.early_stopper.enabled:
                if self.early_stopper(current_metric):
                    print(f"🚨 Early stopping triggered after epoch {epoch} (patience {self.early_stopper.patience}).")
                    break # Stop training loop

        total_training_time = time.time() - total_start_time
        print(f"🏁 Training finished. Total time: {total_training_time:.2f}s")
        print(f"🏆 Best validation metric ({self.early_stopper.metric_key}): {self.best_metric:.4f}")


    def _train_epoch(self, epoch):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        start_time = time.time()

        # Use tqdm for progress bar if installed
        try:
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch} [Train]")
        except ImportError:
            pbar = enumerate(self.train_loader)
            print(f"Epoch {epoch} [Train]")


        for batch_idx, (data, target) in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            # Ensure target shape/type matches output and criterion expectations
            if self.task_type == 'regression' and self.criterion.__class__.__name__ in ['MSELoss', 'L1Loss']:
                 output = output.squeeze(-1) if output.ndim > 1 else output
                 target = target.squeeze(-1) if target.ndim > 1 else target
                 target = target.float() # Ensure target is float for regression loss

            elif self.task_type == 'classification':
                target = target.long() # Ensure target is long for CrossEntropyLoss

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

            # Log batch metrics periodically
            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                batch_time = time.time() - start_time
                # Log batch loss to console/optional tensorboard
                # print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}") # Can be verbose
                if isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})


        avg_loss = total_loss / len(self.train_loader)
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        metrics = calculate_metrics(all_outputs, all_targets, self.task_type)

        return avg_loss, metrics


    def _validate_epoch(self, epoch):
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []

        # Use tqdm for progress bar if installed
        try:
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=f"Epoch {epoch} [Val]")
        except ImportError:
            pbar = enumerate(self.val_loader)
            print(f"Epoch {epoch} [Val]")


        with torch.no_grad():
            for batch_idx, (data, target) in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Ensure target shape/type matches output and criterion expectations
                if self.task_type == 'regression' and self.criterion.__class__.__name__ in ['MSELoss', 'L1Loss']:
                     output = output.squeeze(-1) if output.ndim > 1 else output
                     target = target.squeeze(-1) if target.ndim > 1 else target
                     target = target.float() # Ensure target is float for regression loss
                elif self.task_type == 'classification':
                    target = target.long() # Ensure target is long for CrossEntropyLoss

                loss = self.criterion(output, target)
                total_loss += loss.item()

                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())

                if isinstance(pbar, tqdm):
                   pbar.set_postfix({'loss': f'{loss.item():.4f}'})


        avg_loss = total_loss / len(self.val_loader)
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        metrics = calculate_metrics(all_outputs, all_targets, self.task_type)
        metrics['loss'] = avg_loss # Include loss in validation metrics dict

        return avg_loss, metrics
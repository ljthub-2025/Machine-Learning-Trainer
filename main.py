import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time

from data.dataloader import get_dataloaders
from models import get_model
from engine.trainer import Trainer
from engine.tester import test_model # Import the test function
from utils.logger import Logger
from utils.early_stopping import EarlyStopping
from utils.checkpoint import CheckpointManager

def load_config(config_path):
    """Loads YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        # print(config) # Optional: print loaded config
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device(config_device):
    """Determines the computation device."""
    if config_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif config_device == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device (Apple Silicon GPU)")
        elif torch.backends.mps.is_built():
             print("MPS device not found. Falling back to CPU.")
             device = torch.device("cpu")
        else:
             print("MPS not available because the current PyTorch install was not built with MPS enabled. Falling back to CPU.")
             device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
    return device

def get_optimizer(model, config):
    """Creates an optimizer based on the config."""
    optimizer_name = config.get('optimizer', 'Adam').lower()
    lr = config['learning_rate']
    # Add weight decay, momentum etc. from config if needed
    # weight_decay = config.get('weight_decay', 0)

    if optimizer_name == 'adam':
        print(f"Using Adam optimizer with learning rate {lr}")
        return optim.Adam(model.parameters(), lr=lr) # Add weight_decay=weight_decay if configured
    elif optimizer_name == 'sgd':
        momentum = config.get('sgd_momentum', 0.9) # Example: add momentum config
        print(f"Using SGD optimizer with learning rate {lr} and momentum {momentum}")
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum) # Add weight_decay if configured
    # Add more optimizers here
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_criterion(config):
    """Creates a loss function based on the config."""
    criterion_name = config.get('criterion', 'CrossEntropyLoss').lower()
    task_type = config['task'].lower()

    if task_type == 'classification':
        if criterion_name == 'crossentropyloss':
            print("Using CrossEntropyLoss for classification.")
            return nn.CrossEntropyLoss()
        elif criterion_name == 'bcewithlogitsloss':
             print("Using BCEWithLogitsLoss for binary classification.")
             return nn.BCEWithLogitsLoss() # Good for binary tasks where model outputs logits
        # Add more classification losses
    elif task_type == 'regression':
        if criterion_name == 'mseloss':
            print("Using MSELoss for regression.")
            return nn.MSELoss()
        elif criterion_name == 'l1loss' or criterion_name == 'maeloss':
            print("Using L1Loss (MAE) for regression.")
            return nn.L1Loss()
        # Add more regression losses
    else:
         raise ValueError(f"Unsupported task type '{task_type}' or criterion '{criterion_name}'")

    # Fallback / default if specific match not found but name is valid PyTorch loss
    try:
        criterion_class = getattr(nn, criterion_name.capitalize()) # Attempt to find by name
        print(f"Using {criterion_class.__name__} based on config name.")
        return criterion_class()
    except AttributeError:
         raise ValueError(f"Unsupported criterion: {criterion_name}")


def main():
    parser = argparse.ArgumentParser(description="ML Trainer Framework")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # --- 1. Load Configuration ---
    config = load_config(args.config)

    # --- 2. Setup Environment ---
    run_name = config.get('run_name', f"run_{int(time.time())}")
    set_seed(config['seed'])
    device = get_device(config['device'])

    # --- 3. Initialize Components ---
    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model
    model = get_model(config).to(device)
    # print(model) # Optional: print model summary

    # Optimizer
    optimizer = get_optimizer(model, config)

    # Criterion (Loss Function)
    criterion = get_criterion(config)

    # Logger
    logger = Logger(log_dir=config['log_dir'], run_name=run_name, config=config)

    # Checkpoint Manager
    checkpoint_manager = CheckpointManager(checkpoint_dir=config['checkpoint_dir'], run_name=run_name, config=config)

    # Early Stopping
    early_stopper = EarlyStopping(config=config, verbose=True)

    # --- 4. Initialize Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        early_stopper=early_stopper,
        checkpoint_manager=checkpoint_manager
    )

    # --- 5. Start Training ---
    try:
        trainer.train()
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- 6. Final Testing (Optional: Load best model) ---
        print("\nStarting final evaluation on the test set...")
        best_checkpoint_path = checkpoint_manager.get_best_checkpoint_path()

        if best_checkpoint_path:
            print(f"Loading best model from: {best_checkpoint_path}")
            checkpoint = checkpoint_manager.load_checkpoint(best_checkpoint_path)
            if checkpoint:
                # Re-create model architecture and load state_dict
                test_model_instance = get_model(config).to(device) # Create a fresh instance
                test_model_instance.load_state_dict(checkpoint['model_state_dict'])
                test_model(
                    model=test_model_instance,
                    criterion=criterion,
                    device=device,
                    test_loader=test_loader,
                    config=config,
                    logger=logger # Pass logger to save test results
                )
            else:
                 print("Could not load the best model checkpoint for final testing.")
        else:
            print("No best model checkpoint found. Testing with the final model state (if training completed).")
            # Optionally test with the model state at the end of training (might not be the best)
            test_model(
                    model=model, # Use the model object directly from trainer
                    criterion=criterion,
                    device=device,
                    test_loader=test_loader,
                    config=config,
                    logger=logger # Pass logger to save test results
                )

        # --- 7. Close Logger ---
        logger.close()

if __name__ == '__main__':
    main()
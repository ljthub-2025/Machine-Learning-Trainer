import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression # For synthetic data
import pandas as pd
import numpy as np
import os

from .preprocess import get_image_transforms, preprocess_tabular_data, TabularDataset

def get_dataloaders(config):
    """
    Creates and returns train, validation, and test dataloaders based on the config.
    """
    dataset_name = config['dataset'].lower()
    data_path = config.get('data_path', './data_cache')
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 2)
    task_type = config['task'].lower()

    print(f"Loading dataset: {dataset_name}")

    if dataset_name == 'mnist':
        transform = get_image_transforms(dataset_name)
        # Ensure data path exists
        os.makedirs(data_path, exist_ok=True)

        # Download and load the training data
        train_dataset_full = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

        # Download and load the test data
        test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

        # Split training data into training and validation
        train_size = int(0.8 * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        generator = torch.Generator().manual_seed(config.get('seed', 42)) # For reproducibility
        train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size], generator=generator)

    elif dataset_name == 'synthetic_regression':
        # Create a synthetic dataset for regression
        n_samples = 1000
        n_features = 20
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=config.get('seed', 42))
        y = y.astype(np.float32).reshape(-1, 1) # Ensure correct shape and type for regression
        X = X.astype(np.float32)

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=config.get('seed', 42))
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=config.get('seed', 42))

        # Create TensorDatasets
        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        test_dataset = TabularDataset(X_test, y_test)
        # Adjust target dtype after creation if needed
        train_dataset.targets = train_dataset.targets.float()
        val_dataset.targets = val_dataset.targets.float()
        test_dataset.targets = test_dataset.targets.float()


    # --- Example: Loading a custom tabular dataset from CSV ---
    # elif dataset_name == 'your_custom_tabular':
    #     try:
    #         # Assumes you have train.csv, val.csv, test.csv in data_path
    #         df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    #         df_val = pd.read_csv(os.path.join(data_path, 'val.csv'))
    #         df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"CSV files not found in {data_path} for dataset '{dataset_name}'")

    #     target_column = config.get('target_column', 'target') # Specify target column in config
    #     if target_column not in df_train.columns:
    #          raise ValueError(f"Target column '{target_column}' not found in training data.")

    #     X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_tabular_data(
    #         df_train, df_val, df_test, target_column
    #     )

    #     train_dataset = TabularDataset(X_train, y_train)
    #     val_dataset = TabularDataset(X_val, y_val)
    #     test_dataset = TabularDataset(X_test, y_test)

    #     # Adjust target dtype based on task AFTER TabularDataset creation
    #     if task_type == 'regression':
    #         train_dataset.targets = train_dataset.targets.float()
    #         val_dataset.targets = val_dataset.targets.float()
    #         test_dataset.targets = test_dataset.targets.float()
    #     elif task_type == 'classification':
    #         # Assuming integer labels for classification
    #         train_dataset.targets = train_dataset.targets.long()
    #         val_dataset.targets = val_dataset.targets.long()
    #         test_dataset.targets = test_dataset.targets.long()

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Data loading complete. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    # Print sample shape for debugging
    try:
        sample_inputs, sample_targets = next(iter(train_loader))
        print(f"Sample input batch shape: {sample_inputs.shape}")
        print(f"Sample target batch shape: {sample_targets.shape}")
        print(f"Sample target dtype: {sample_targets.dtype}")
    except StopIteration:
        print("Could not retrieve sample batch (dataset might be empty).")


    return train_loader, val_loader, test_loader
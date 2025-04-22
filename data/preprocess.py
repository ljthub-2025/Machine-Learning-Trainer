import torch
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# --- Image Preprocessing ---

def get_image_transforms(dataset_name):
    """Returns appropriate image transforms."""
    if dataset_name.lower() == 'mnist':
        # MNIST specific transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific mean/std
        ])
    elif dataset_name.lower() == 'cifar10':
         # CIFAR-10 specific transforms (example)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    # Add more dataset transforms here
    else:
        # Default / Generic
        return transforms.Compose([
            transforms.ToTensor(),
            # Add generic normalization if applicable, otherwise just ToTensor
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Example for general images
        ])

# --- Tabular Preprocessing ---

def preprocess_tabular_data(df_train, df_val, df_test, target_column):
    """Scales features and separates target."""
    features = [col for col in df_train.columns if col != target_column]

    scaler = StandardScaler()

    # Fit on training data only
    df_train_scaled = df_train.copy()
    df_train_scaled[features] = scaler.fit_transform(df_train[features])

    # Transform validation and test data
    df_val_scaled = df_val.copy()
    df_val_scaled[features] = scaler.transform(df_val[features])

    df_test_scaled = df_test.copy()
    df_test_scaled[features] = scaler.transform(df_test[features])

    X_train = df_train_scaled[features].values.astype(np.float32)
    y_train = df_train_scaled[target_column].values

    X_val = df_val_scaled[features].values.astype(np.float32)
    y_val = df_val_scaled[target_column].values

    X_test = df_test_scaled[features].values.astype(np.float32)
    y_test = df_test_scaled[target_column].values

    # Adjust target dtype for regression/classification
    # Assuming classification uses integer labels and regression uses float targets
    # This might need adjustment based on the specific loss function
    if y_train.dtype == 'object' or pd.api.types.is_integer_dtype(y_train): # Simple check
         y_train = y_train.astype(np.int64)
         y_val = y_val.astype(np.int64)
         y_test = y_test.astype(np.int64)
    else:
         y_train = y_train.astype(np.float32).reshape(-1, 1)
         y_val = y_val.astype(np.float32).reshape(-1, 1)
         y_test = y_test.astype(np.float32).reshape(-1, 1)


    return X_train, y_train, X_val, y_val, X_test, y_test, scaler # Return scaler if needed later

class TabularDataset(torch.utils.data.Dataset):
    """Simple dataset for tabular data."""
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Ensure targets have the correct type based on task later (in get_dataloaders)
        self.targets = torch.tensor(targets) # Type adjusted later

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
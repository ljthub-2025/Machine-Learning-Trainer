import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size (int): Dimension of input features.
            hidden_sizes (list of int): List containing the size of each hidden layer.
            output_size (int): Dimension of the output.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Add activation function for all layers except the last one
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
                # Optionally add Dropout
                # self.layers.append(nn.Dropout(p=0.5))

    def forward(self, x):
        """Forward pass"""
        # Flatten input if it's not already flat (e.g., coming from image data)
        if x.dim() > 2:
           x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x
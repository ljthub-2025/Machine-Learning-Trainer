# This file can be empty or used for package initialization
from .mlp import MLP
from .cnn import SimpleCNN

def get_model(config):
    """Factory function to create a model based on the config."""
    model_name = config['model'].lower()
    task_type = config['task'].lower()

    if model_name == 'mlp':
        params = config.get('mlp', {}) # Get MLP specific params
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_sizes = params.get('hidden_sizes', [128, 64]) # Default hidden layers

        if input_size is None or output_size is None:
            raise ValueError("MLP requires 'input_size' and 'output_size' in config['mlp']")

        print(f"Creating MLP model with input: {input_size}, hidden: {hidden_sizes}, output: {output_size}")
        return MLP(input_size, hidden_sizes, output_size)

    elif model_name == 'cnn':
        params = config.get('cnn', {}) # Get CNN specific params
        input_channels = params.get('input_channels')
        num_classes = params.get('num_classes') # Often same as output_size for classification

        if input_channels is None or num_classes is None:
             raise ValueError("CNN requires 'input_channels' and 'num_classes' in config['cnn']")

        print(f"Creating SimpleCNN model with input channels: {input_channels}, num_classes: {num_classes}")
        return SimpleCNN(input_channels, num_classes)

    # Add more models here
    # elif model_name == 'resnet':
    #     from torchvision.models import resnet18
    #     # Example: Modify torchvision model
    #     model = resnet18(pretrained=config.get('pretrained', False))
    #     num_ftrs = model.fc.in_features
    #     output_size = config[model_name].get('output_size') # Get from config
    #     model.fc = torch.nn.Linear(num_ftrs, output_size)
    #     return model

    else:
        raise ValueError(f"Unsupported model type: {model_name}")
import torch
from tqdm import tqdm
import numpy as np

from utils.metrics import calculate_metrics

def test_model(model, criterion, device, test_loader, config, logger):
    """
    Evaluates the model on the test dataset.
    """
    print("\nEvaluating on test set...")
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    task_type = config['task'].lower()

    # Use tqdm for progress bar if installed
    try:
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="[Test]")
    except ImportError:
        pbar = enumerate(test_loader)
        print("[Test]")

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Ensure target shape/type matches output and criterion expectations
            if task_type == 'regression' and criterion.__class__.__name__ in ['MSELoss', 'L1Loss']:
                output = output.squeeze(-1) if output.ndim > 1 else output
                target = target.squeeze(-1) if target.ndim > 1 else target
                target = target.float() # Ensure target is float for regression loss
            elif task_type == 'classification':
                target = target.long() # Ensure target is long for CrossEntropyLoss

            loss = criterion(output, target)
            total_loss += loss.item()

            all_outputs.append(output.detach().cpu())
            all_targets.append(target.detach().cpu())

            if isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})


    avg_loss = total_loss / len(test_loader)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    metrics = calculate_metrics(all_outputs, all_targets, task_type)
    metrics['loss'] = avg_loss # Include loss in test metrics dict

    print(f"🧪 Test Results: Loss: {avg_loss:.4f} | Metrics: {metrics}")

    # Log test metrics
    test_log_metrics = {f"test_{k}": v for k, v in metrics.items()}
    test_log_metrics["test_loss"] = avg_loss
    logger.log_metrics(step=0, metrics=test_log_metrics, phase='test') # Step 0 for final test results

    return avg_loss, metrics
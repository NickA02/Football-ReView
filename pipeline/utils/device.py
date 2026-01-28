"""Device resolution utilities."""
import torch


def resolve_device(device: str) -> str:
    """Resolve device string to appropriate PyTorch device.
    
    Args:
        device: 'auto', 'cpu', 'mps', or cuda device string like '0'
        
    Returns:
        Resolved device string
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "0"
        else:
            return "cpu"
    return device

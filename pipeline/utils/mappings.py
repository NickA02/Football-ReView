"""Class ID to yard mapping utilities."""
from typing import Optional


def cls_to_yards(cls_id: int) -> Optional[int]:
    """Map yard number class ID to actual yard value.
    
    Update this mapping if your dataset uses different IDs.
    
    Args:
        cls_id: Class ID from detection model
        
    Returns:
        Yard value (10, 20, 30, 40, 50) or None if invalid
    """
    mapping = {
        0: 10,
        1: 20,
        2: 30,
        3: 40,
        4: 50,
    }
    return mapping.get(cls_id)

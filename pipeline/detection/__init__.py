"""Detection modules for field elements and players."""
from .field_detection import detect_hash_and_numbers
from .player_detection import detect_players_and_poses

__all__ = ["detect_hash_and_numbers", "detect_players_and_poses"]

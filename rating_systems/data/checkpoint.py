"""Checkpoint save/load for batch rating systems (WHR, TTT).

Stores fitted state as a single .npz file with embedded JSON metadata.
Supports incremental-append detection via data fingerprinting.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def compute_fingerprint(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    num_players: int,
    max_day: int,
) -> dict:
    """Compute a fingerprint of the dataset for checkpoint validation.

    Args:
        player1: Player 1 IDs array.
        player2: Player 2 IDs array.
        scores: Scores array.
        num_players: Total number of players.
        max_day: Maximum day index.

    Returns:
        Dict with num_games, num_players, max_day, data_hash.
    """
    n = len(player1)
    h = hashlib.sha256()
    h.update(player1[:n].tobytes())
    h.update(player2[:n].tobytes())
    h.update(scores[:n].tobytes())
    return {
        "num_games": n,
        "num_players": num_players,
        "max_day": int(max_day),
        "data_hash": h.hexdigest(),
    }


def verify_fingerprint(
    player1: np.ndarray,
    player2: np.ndarray,
    scores: np.ndarray,
    fingerprint: dict,
) -> bool:
    """Check that a new dataset is a compatible superset of the checkpoint.

    The first ``fingerprint['num_games']`` games must match exactly.

    Args:
        player1: New dataset player 1 IDs.
        player2: New dataset player 2 IDs.
        scores: New dataset scores.
        fingerprint: Fingerprint from the checkpoint.

    Returns:
        True if the new dataset's prefix matches the checkpoint.
    """
    n = fingerprint["num_games"]
    if len(player1) < n:
        return False
    h = hashlib.sha256()
    h.update(player1[:n].tobytes())
    h.update(player2[:n].tobytes())
    h.update(scores[:n].tobytes())
    return h.hexdigest() == fingerprint["data_hash"]


def save_checkpoint(
    path: str,
    arrays: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    """Save arrays and metadata to a single .npz file.

    Args:
        path: Output file path.
        arrays: Dict of name -> numpy array.
        metadata: JSON-serialisable metadata dict.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta_bytes = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    np.savez(str(p), _metadata=meta_bytes, **arrays)


def load_checkpoint(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load a checkpoint from a .npz file.

    Args:
        path: Input file path.

    Returns:
        Tuple of (arrays_dict, metadata_dict).
    """
    data = np.load(str(path), allow_pickle=False)
    meta_bytes = data["_metadata"].tobytes()
    metadata = json.loads(meta_bytes.decode("utf-8"))
    arrays = {k: data[k] for k in data.files if k != "_metadata"}
    return arrays, metadata

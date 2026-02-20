"""Directed weighted match graph with time decay for Hodge decomposition."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


class MatchGraph:
    """Directed weighted match graph with time decay.

    Stores match results as (winner, loser, day) triples and provides
    efficient queries for common opponents and subgraph extraction.
    """

    def __init__(self) -> None:
        # (winner, loser) -> list of match days
        self._wins: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        # player -> set of opponents (either direction)
        self._opponents: Dict[int, Set[int]] = defaultdict(set)

    def add_result(self, winner: int, loser: int, day: int) -> None:
        """Record a match result."""
        self._wins[(winner, loser)].append(day)
        self._opponents[winner].add(loser)
        self._opponents[loser].add(winner)

    def get_common_opponents(self, p1: int, p2: int) -> Set[int]:
        """Return the set of players that both p1 and p2 have played."""
        return (self._opponents.get(p1, set()) & self._opponents.get(p2, set())) - {p1, p2}

    def get_expanded_nodes(
        self, p1: int, p2: int, max_nodes: int = 50
    ) -> List[int]:
        """Return an expanded subgraph node list for p1 vs p2.

        Prioritizes common opponents, then fills from the union of both
        players' opponents ranked by connectivity (number of opponents).
        Always starts with [p1, p2].

        Args:
            p1: Player 1 ID.
            p2: Player 2 ID.
            max_nodes: Maximum number of nodes in the subgraph.

        Returns:
            List of node IDs starting with [p1, p2], length <= max_nodes.
        """
        common = (self._opponents.get(p1, set()) & self._opponents.get(p2, set())) - {p1, p2}
        union = (self._opponents.get(p1, set()) | self._opponents.get(p2, set())) - {p1, p2}

        budget = max_nodes - 2  # reserve slots for p1, p2

        # Common opponents always included first
        common_sorted = sorted(common)
        selected: List[int] = common_sorted[:budget]

        if len(selected) < budget:
            # Fill remaining slots from union (excluding already-selected common)
            remaining = union - common
            # Rank by connectivity (number of opponents) descending
            remaining_ranked = sorted(
                remaining, key=lambda p: len(self._opponents.get(p, set())), reverse=True
            )
            selected.extend(remaining_ranked[: budget - len(selected)])

        return [p1, p2] + selected

    def get_subgraph_flows(
        self,
        nodes: List[int],
        current_day: int,
        half_life: float,
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Extract time-decayed net flows for a subgraph.

        Args:
            nodes: List of node IDs defining the subgraph.
            current_day: Reference day for time decay.
            half_life: Half-life in days for exponential decay.

        Returns:
            (node_list, edge_flows, edge_weights) where:
            - node_list: the input nodes (for index mapping)
            - edge_flows: (E,3) array of [i_idx, j_idx, net_flow] for each
              directed edge (i < j in node_list order)
            - edge_weights: (E,) total weight per edge
        """
        node_set = set(nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        decay_rate = np.log(2) / half_life

        edges = []
        for i_pos, ni in enumerate(nodes):
            for j_pos in range(i_pos + 1, len(nodes)):
                nj = nodes[j_pos]

                # Compute weighted wins in each direction
                flow_ij = 0.0  # ni beat nj
                flow_ji = 0.0  # nj beat ni
                total_weight = 0.0

                for day in self._wins.get((ni, nj), []):
                    w = np.exp(-decay_rate * (current_day - day))
                    flow_ij += w
                    total_weight += w

                for day in self._wins.get((nj, ni), []):
                    w = np.exp(-decay_rate * (current_day - day))
                    flow_ji += w
                    total_weight += w

                if total_weight > 0:
                    net_flow = flow_ij - flow_ji
                    edges.append((i_pos, j_pos, net_flow, total_weight))

        if not edges:
            return nodes, np.empty((0, 3)), np.empty(0)

        edge_arr = np.array(edges)
        edge_flows = edge_arr[:, :3]  # [i_idx, j_idx, net_flow]
        edge_weights = edge_arr[:, 3]  # total_weight

        return nodes, edge_flows, edge_weights

    def save_arrays(self) -> Dict[str, np.ndarray]:
        """Serialize graph to numpy arrays for checkpointing."""
        if not self._wins:
            return {
                "graph_win_from": np.array([], dtype=np.int64),
                "graph_win_to": np.array([], dtype=np.int64),
                "graph_win_day": np.array([], dtype=np.int32),
            }

        win_from = []
        win_to = []
        win_day = []
        for (w, l), days in self._wins.items():
            for d in days:
                win_from.append(w)
                win_to.append(l)
                win_day.append(d)

        return {
            "graph_win_from": np.array(win_from, dtype=np.int64),
            "graph_win_to": np.array(win_to, dtype=np.int64),
            "graph_win_day": np.array(win_day, dtype=np.int32),
        }

    def load_arrays(self, arrays: Dict[str, np.ndarray]) -> None:
        """Restore graph from checkpoint arrays."""
        self._wins.clear()
        self._opponents.clear()

        win_from = arrays["graph_win_from"]
        win_to = arrays["graph_win_to"]
        win_day = arrays["graph_win_day"]

        for i in range(len(win_from)):
            w = int(win_from[i])
            l = int(win_to[i])
            d = int(win_day[i])
            self._wins[(w, l)].append(d)
            self._opponents[w].add(l)
            self._opponents[l].add(w)

    def __len__(self) -> int:
        """Number of recorded match results."""
        return sum(len(days) for days in self._wins.values())

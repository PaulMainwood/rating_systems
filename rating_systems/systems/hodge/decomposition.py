"""Hodge decomposition for intransitivity measurement.

Implements the Hodge decomposition on a weighted directed graph following
Jiang et al. (2011) "Statistical ranking and combinatorial Hodge theory".

The key idea: any edge flow f on a graph can be decomposed into
  f = f_grad + f_curl + f_harm
where f_grad is the transitive (gradient) component and f_curl captures
cyclic (intransitive) patterns. The ratio ||f_curl|| / ||f|| measures
how much of the pairwise comparison data is intransitive.
"""

import numpy as np

from .graph import MatchGraph


def hodge_intransitivity(
    n_nodes: int,
    edge_flows: np.ndarray,
    edge_weights: np.ndarray,
) -> float:
    """Compute intransitivity score I for a subgraph via Hodge decomposition.

    Args:
        n_nodes: Number of nodes in the subgraph.
        edge_flows: (E, 3) array of [i_idx, j_idx, net_flow].
        edge_weights: (E,) total weight per edge.

    Returns:
        Intransitivity ratio I in [0, 1], or NaN if no edges.
    """
    n_edges = len(edge_flows)
    if n_edges == 0 or n_nodes < 2:
        return np.nan

    i_idx = edge_flows[:, 0].astype(int)
    j_idx = edge_flows[:, 1].astype(int)
    flows = edge_flows[:, 2]

    # Build weighted graph Laplacian L_w (n x n) — vectorized
    L_w = np.zeros((n_nodes, n_nodes))
    np.add.at(L_w, (i_idx, i_idx), edge_weights)
    np.add.at(L_w, (j_idx, j_idx), edge_weights)
    np.add.at(L_w, (i_idx, j_idx), -edge_weights)
    np.add.at(L_w, (j_idx, i_idx), -edge_weights)

    # Build divergence vector δ = d₀ᵀ W f (n,) — vectorized
    div = np.zeros(n_nodes)
    wf = edge_weights * flows
    np.add.at(div, i_idx, -wf)
    np.add.at(div, j_idx, wf)

    # Solve L_w * s = δ for node potentials s (lstsq since L is singular)
    s, _, _, _ = np.linalg.lstsq(L_w, div, rcond=None)

    # Gradient flow: f_grad(i,j) = s[j] - s[i]
    f_grad = s[j_idx] - s[i_idx]

    # Curl flow: f_curl = f - f_grad
    f_curl = flows - f_grad

    # Weighted norms
    total_norm_sq = np.sum(edge_weights * flows ** 2)
    curl_norm_sq = np.sum(edge_weights * f_curl ** 2)

    if total_norm_sq < 1e-15:
        return 0.0

    return float(np.sqrt(curl_norm_sq / total_norm_sq))


def compute_i_star(
    p1: int,
    p2: int,
    graph: MatchGraph,
    current_day: int,
    half_life: float,
    min_common: int = 2,
    max_nodes: int = 50,
) -> float:
    """Compute evidence-weighted intransitivity score I* for a matchup.

    When there are enough common opponents (>= min_common), uses the original
    subgraph of {p1, p2} + common opponents weighted by sqrt(|common|).

    When there are fewer than min_common common opponents, falls back to an
    expanded subgraph built from the union of both players' opponents (capped
    at max_nodes). The score is weighted by sqrt(n_bridging) where n_bridging
    is the number of nodes connected to both p1 and p2 in the subgraph.

    Args:
        p1: Player 1 ID.
        p2: Player 2 ID.
        graph: MatchGraph with historical results.
        current_day: Current day for time decay.
        half_life: Half-life in days for exponential decay.
        min_common: Minimum common opponents required for direct approach.
        max_nodes: Maximum subgraph size for expanded fallback.

    Returns:
        I* = I * sqrt(evidence), or NaN if insufficient data.
    """
    common = graph.get_common_opponents(p1, p2)

    if len(common) >= min_common:
        # Original approach: subgraph of {p1, p2} + common opponents
        common_list = sorted(common)
        if len(common_list) > max_nodes - 2:
            common_list = common_list[: max_nodes - 2]
        nodes = [p1, p2] + common_list

        _, edge_flows, edge_weights = graph.get_subgraph_flows(
            nodes, current_day, half_life
        )
        if len(edge_flows) == 0:
            return np.nan

        I = hodge_intransitivity(len(nodes), edge_flows, edge_weights)
        if np.isnan(I):
            return np.nan

        return float(I * np.sqrt(len(common_list)))

    # Expanded fallback: union of opponents, prioritized by connectivity
    nodes = graph.get_expanded_nodes(p1, p2, max_nodes)
    if len(nodes) <= 2:
        return np.nan

    # Count bridging nodes: subgraph nodes (other than p1/p2) connected to BOTH
    node_set = set(nodes) - {p1, p2}
    p1_opps = graph._opponents.get(p1, set())
    p2_opps = graph._opponents.get(p2, set())
    n_bridging = len(node_set & p1_opps & p2_opps)

    if n_bridging == 0:
        return np.nan

    _, edge_flows, edge_weights = graph.get_subgraph_flows(
        nodes, current_day, half_life
    )
    if len(edge_flows) == 0:
        return np.nan

    I = hodge_intransitivity(len(nodes), edge_flows, edge_weights)
    if np.isnan(I):
        return np.nan

    return float(I * np.sqrt(n_bridging))

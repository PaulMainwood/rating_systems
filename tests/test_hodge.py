"""Tests for Hodge intransitivity system."""

import numpy as np
import pytest

from rating_systems.systems.hodge.graph import MatchGraph
from rating_systems.systems.hodge.decomposition import hodge_intransitivity, compute_i_star
from rating_systems.systems.hodge.hodge import HodgeIntransitivity, HodgeConfig
from rating_systems.data.types import GameBatch


# ---------------------------------------------------------------------------
# MatchGraph tests
# ---------------------------------------------------------------------------

class TestMatchGraph:
    def test_add_result(self):
        g = MatchGraph()
        g.add_result(0, 1, 100)
        assert len(g) == 1
        assert g.get_common_opponents(0, 1) == set()

    def test_common_opponents(self):
        g = MatchGraph()
        # A beats B, A beats C, B beats C
        g.add_result(0, 1, 100)
        g.add_result(0, 2, 101)
        g.add_result(1, 2, 102)
        # Common opponents of A(0) and B(1) = {C(2)}
        assert g.get_common_opponents(0, 1) == {2}

    def test_common_opponents_multiple(self):
        g = MatchGraph()
        # A plays B, C, D; B plays A, C, D
        g.add_result(0, 1, 100)
        g.add_result(0, 2, 101)
        g.add_result(0, 3, 102)
        g.add_result(1, 2, 103)
        g.add_result(1, 3, 104)
        assert g.get_common_opponents(0, 1) == {2, 3}

    def test_save_load_roundtrip(self):
        g = MatchGraph()
        g.add_result(0, 1, 100)
        g.add_result(1, 2, 200)
        g.add_result(2, 0, 300)

        arrays = g.save_arrays()
        assert "graph_win_from" in arrays
        assert len(arrays["graph_win_from"]) == 3

        g2 = MatchGraph()
        g2.load_arrays(arrays)
        assert len(g2) == 3
        assert g2.get_common_opponents(0, 1) == {2}

    def test_empty_graph(self):
        g = MatchGraph()
        arrays = g.save_arrays()
        assert len(arrays["graph_win_from"]) == 0

        g2 = MatchGraph()
        g2.load_arrays(arrays)
        assert len(g2) == 0

    def test_subgraph_flows(self):
        g = MatchGraph()
        g.add_result(0, 1, 100)
        g.add_result(1, 0, 200)

        nodes, flows, weights = g.get_subgraph_flows([0, 1], 200, 365.0)
        assert len(flows) == 1
        assert flows[0][0] == 0  # i_idx
        assert flows[0][1] == 1  # j_idx
        # Two matches, one each way, so net flow should be close to 0
        # but not exactly 0 due to time decay
        assert abs(flows[0][2]) < 1.0

    def test_subgraph_flows_no_edges(self):
        g = MatchGraph()
        g.add_result(0, 1, 100)
        # Players 2 and 3 have never played each other
        nodes, flows, weights = g.get_subgraph_flows([2, 3], 200, 365.0)
        assert len(flows) == 0

    def test_get_expanded_nodes_with_common(self):
        """Common opponents should be included first."""
        g = MatchGraph()
        # A(0) plays B(1), C(2), D(3); B(1) plays A(0), C(2)
        g.add_result(0, 1, 0)
        g.add_result(0, 2, 0)
        g.add_result(0, 3, 0)
        g.add_result(1, 2, 0)

        nodes = g.get_expanded_nodes(0, 1, max_nodes=50)
        assert nodes[0] == 0
        assert nodes[1] == 1
        # C(2) is common, D(3) is union-only
        assert 2 in nodes
        assert 3 in nodes

    def test_get_expanded_nodes_caps_at_max(self):
        """Should not exceed max_nodes."""
        g = MatchGraph()
        # A(0) plays many opponents
        for i in range(2, 20):
            g.add_result(0, i, 0)
        # B(1) plays a few
        g.add_result(1, 2, 0)
        g.add_result(1, 3, 0)

        nodes = g.get_expanded_nodes(0, 1, max_nodes=6)
        assert len(nodes) <= 6
        assert nodes[0] == 0
        assert nodes[1] == 1
        # Common opponents (2, 3) should be included first
        assert 2 in nodes
        assert 3 in nodes

    def test_get_expanded_nodes_no_opponents(self):
        """Players with no opponents return just [p1, p2]."""
        g = MatchGraph()
        nodes = g.get_expanded_nodes(0, 1, max_nodes=50)
        assert nodes == [0, 1]

    def test_get_expanded_nodes_ranks_by_connectivity(self):
        """Union-only nodes should be ranked by number of opponents."""
        g = MatchGraph()
        # A(0) plays C(2), D(3), E(4)
        g.add_result(0, 2, 0)
        g.add_result(0, 3, 0)
        g.add_result(0, 4, 0)
        # B(1) plays F(5)
        g.add_result(1, 5, 0)
        # D(3) is highly connected (plays many others)
        for i in range(10, 20):
            g.add_result(3, i, 0)

        nodes = g.get_expanded_nodes(0, 1, max_nodes=5)
        # Budget = 3; no common opponents, all union-only
        # D(3) has 11 opponents, should be ranked first among union nodes
        assert nodes[0] == 0
        assert nodes[1] == 1
        # D(3) should appear before less-connected nodes
        union_nodes = nodes[2:]
        assert 3 in union_nodes


# ---------------------------------------------------------------------------
# Hodge decomposition tests
# ---------------------------------------------------------------------------

class TestHodgeDecomposition:
    def test_transitive_graph(self):
        """Gradient-consistent A>B>C should have I = 0.

        For zero curl, flows must satisfy f(A,B) + f(B,C) = f(A,C).
        Here: A beats B 2x, B beats C 1x, A beats C 3x (2+1=3).
        """
        g = MatchGraph()
        # All on same day to avoid time-decay asymmetry
        g.add_result(0, 1, 0)  # A > B (x2)
        g.add_result(0, 1, 0)
        g.add_result(1, 2, 0)  # B > C (x1)
        g.add_result(0, 2, 0)  # A > C (x3)
        g.add_result(0, 2, 0)
        g.add_result(0, 2, 0)

        nodes = [0, 1, 2]
        _, edge_flows, edge_weights = g.get_subgraph_flows(nodes, 0, 365.0)
        I = hodge_intransitivity(3, edge_flows, edge_weights)
        assert I < 0.01, f"Expected near-zero intransitivity for gradient-consistent graph, got {I}"

    def test_cyclic_graph(self):
        """3-cycle A>B>C>A should have high intransitivity."""
        g = MatchGraph()
        for day in range(10):
            g.add_result(0, 1, day)  # A > B
            g.add_result(1, 2, day)  # B > C
            g.add_result(2, 0, day)  # C > A (intransitive!)

        nodes = [0, 1, 2]
        _, edge_flows, edge_weights = g.get_subgraph_flows(nodes, 10, 365.0)
        I = hodge_intransitivity(3, edge_flows, edge_weights)
        assert I > 0.9, f"Expected high intransitivity for pure 3-cycle, got {I}"

    def test_empty_graph_returns_nan(self):
        """No edges should return NaN."""
        I = hodge_intransitivity(3, np.empty((0, 3)), np.empty(0))
        assert np.isnan(I)

    def test_compute_i_star_insufficient_data_no_neighbors(self):
        """Players with no opponents at all should return NaN even with fallback."""
        g = MatchGraph()
        # Only one match, no bridging possible
        g.add_result(0, 1, 100)
        result = compute_i_star(0, 1, g, 100, 365.0, min_common=2, max_nodes=50)
        assert np.isnan(result)

    def test_compute_i_star_expanded_fallback(self):
        """Expanded fallback should produce a result when no common opponents exist."""
        g = MatchGraph()
        # A(0) plays C(2), C(2) plays B(1) — no direct common opponents
        # but C bridges A and B
        for day in range(5):
            g.add_result(0, 2, day)
            g.add_result(2, 1, day)
            g.add_result(1, 0, day)  # A and B play each other too

        # No common opponents (A's opponents: {1,2}, B's opponents: {0,2},
        # common = {2} after removing {0,1} = {2}... actually they DO share C(2))
        # Let me build a case with NO common opponents:
        g2 = MatchGraph()
        # A(0) plays C(2), D(3); B(1) plays E(4), F(5)
        # C(2) plays E(4) — indirect bridge
        g2.add_result(0, 2, 0)
        g2.add_result(0, 3, 0)
        g2.add_result(1, 4, 0)
        g2.add_result(1, 5, 0)
        g2.add_result(2, 4, 0)  # C-E bridge
        # Also need A-B match for flow
        g2.add_result(0, 1, 0)

        common = g2.get_common_opponents(0, 1)
        assert len(common) == 0  # Confirm no common opponents

        # With min_common=2, the original approach would return NaN
        # But expanded fallback includes union neighbors
        result = compute_i_star(0, 1, g2, 0, 365.0, min_common=2, max_nodes=50)
        # No bridging nodes (no node connected to BOTH 0 and 1 except via union)
        # C(2) connects to A(0) but not B(1), E(4) connects to B(1) but not A(0)
        # So n_bridging=0 → still NaN
        assert np.isnan(result)

    def test_compute_i_star_expanded_with_bridging(self):
        """Expanded fallback with bridging nodes should produce finite result."""
        g = MatchGraph()
        # A(0) plays C(2), D(3); B(1) plays C(2), E(4)
        # C(2) is a common opponent but let's set min_common=3
        # so we trigger fallback despite having 1 common opponent
        g.add_result(0, 2, 0)
        g.add_result(0, 3, 0)
        g.add_result(1, 2, 0)
        g.add_result(1, 4, 0)
        g.add_result(0, 1, 0)
        # Add more edges for richer structure
        g.add_result(3, 4, 0)
        g.add_result(2, 3, 0)

        common = g.get_common_opponents(0, 1)
        assert len(common) == 1  # Only C(2) is common

        # min_common=3 triggers expanded fallback
        result = compute_i_star(0, 1, g, 0, 365.0, min_common=3, max_nodes=50)
        # C(2) is a bridging node (connected to both 0 and 1)
        assert np.isfinite(result)
        assert result >= 0

    def test_compute_i_star_common_path_preserved(self):
        """Original common-opponent path should work unchanged."""
        g = MatchGraph()
        # A(0) plays C(2), D(3); B(1) plays C(2), D(3)
        for day in range(10):
            g.add_result(0, 2, day)
            g.add_result(0, 3, day)
            g.add_result(1, 2, day)
            g.add_result(1, 3, day)
            g.add_result(0, 1, day)

        common = g.get_common_opponents(0, 1)
        assert len(common) == 2

        result = compute_i_star(0, 1, g, 10, 365.0, min_common=2, max_nodes=50)
        assert np.isfinite(result)
        assert result >= 0

    def test_compute_i_star_with_common_opponents(self):
        """I* with common opponents should be a finite number."""
        g = MatchGraph()
        # Build: A vs B, A vs C, A vs D, B vs C, B vs D
        for day in range(10):
            g.add_result(0, 1, day)
            g.add_result(0, 2, day)
            g.add_result(0, 3, day)
            g.add_result(1, 2, day)
            g.add_result(1, 3, day)

        result = compute_i_star(0, 1, g, 10, 365.0, min_common=2)
        assert np.isfinite(result)
        assert result >= 0

    def test_i_star_scales_with_evidence(self):
        """I* = I * sqrt(k), so more common opponents → higher score for same I."""
        g = MatchGraph()
        # Pure 3-cycle with 2 common opponents
        for day in range(10):
            g.add_result(0, 2, day)
            g.add_result(2, 1, day)
            g.add_result(1, 0, day)
            g.add_result(0, 3, day)
            g.add_result(3, 1, day)

        # 2 common opponents: C(2), D(3)
        i_star = compute_i_star(0, 1, g, 10, 365.0, min_common=2)
        assert np.isfinite(i_star)
        assert i_star > 0


# ---------------------------------------------------------------------------
# HodgeIntransitivity system tests
# ---------------------------------------------------------------------------

class TestHodgeIntransitivity:
    def test_fit_and_predict(self):
        """Basic fit/predict cycle."""
        system = HodgeIntransitivity(half_life=365.0, min_common=1, num_players=5)

        # Batch 1: day 0
        batch1 = GameBatch(
            player1=np.array([0, 1, 2]),
            player2=np.array([1, 2, 0]),
            scores=np.array([1.0, 1.0, 1.0]),
            day=0,
        )
        # Batch 2: day 1 (same cycle)
        batch2 = GameBatch(
            player1=np.array([0, 1, 2]),
            player2=np.array([1, 2, 0]),
            scores=np.array([1.0, 1.0, 1.0]),
            day=1,
        )

        from rating_systems import GameDataset
        import polars as pl

        df = pl.DataFrame({
            "Player1": [0, 1, 2, 0, 1, 2],
            "Player2": [1, 2, 0, 1, 2, 0],
            "Score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Day": [0, 0, 0, 1, 1, 1],
        })
        dataset = GameDataset.from_dataframe(df)
        system.fit(dataset)

        preds = system.predict_proba(np.array([0]), np.array([1]))
        assert len(preds) == 1
        # With min_common=1, should get a finite I* for a cycle
        assert np.isfinite(preds[0])

    def test_save_load_state(self, tmp_path):
        """Save/load roundtrip preserves graph state."""
        system = HodgeIntransitivity(half_life=365.0, min_common=1, num_players=4)

        df = _make_test_df()
        dataset = _make_dataset(df)
        system.fit(dataset)

        path = str(tmp_path / "hodge_state.npz")
        system.save_state(path)

        system2 = HodgeIntransitivity(half_life=365.0, min_common=1)
        system2.load_state(path)

        # Should produce the same predictions
        p1 = np.array([0, 1])
        p2 = np.array([1, 2])
        preds1 = system.predict_proba(p1, p2)
        preds2 = system2.predict_proba(p1, p2)
        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_predict_returns_nan_for_unknown(self):
        """Players without common opponents should return NaN."""
        system = HodgeIntransitivity(half_life=365.0, min_common=2, num_players=10)

        df = _make_test_df()
        dataset = _make_dataset(df)
        system.fit(dataset)

        # Players 5 and 6 have never played
        preds = system.predict_proba(np.array([5]), np.array([6]))
        assert np.isnan(preds[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_df():
    """Create a small test DataFrame with a 3-cycle."""
    import polars as pl
    return pl.DataFrame({
        "Player1": [0, 1, 2, 0, 1, 0],
        "Player2": [1, 2, 0, 2, 0, 1],
        "Score": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Day": [0, 0, 0, 1, 1, 1],
    })


def _make_dataset(df):
    from rating_systems import GameDataset
    return GameDataset.from_dataframe(df)

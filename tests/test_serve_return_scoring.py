"""The tennis scoring formulas: point -> game -> tie-break -> set -> match.

Which player is called Player 1 is arbitrary, so the two players' chances
must add up to one and equal players must come out level.
"""

import itertools

import pytest

from rating_systems.systems.serve_return._scoring import (
    prob_hold_serve, prob_win_match, prob_win_set, prob_win_tiebreak,
)

SERVE = [0.50, 0.55, 0.62, 0.64, 0.70, 0.80]


def test_hold_serve_matches_the_textbook_value():
    # A server who wins 60% of points holds 73.6% of games.
    assert prob_hold_serve(0.60) == pytest.approx(0.7357, abs=1e-4)
    assert prob_hold_serve(0.50) == pytest.approx(0.5)


@pytest.mark.parametrize("a,b", itertools.product(SERVE, SERVE))
def test_tiebreak_chances_add_up_to_one(a, b):
    assert prob_win_tiebreak(a, b) + prob_win_tiebreak(b, a) == pytest.approx(1.0, abs=1e-12)


def test_a_tiebreak_between_equals_is_even():
    for p in SERVE:
        assert prob_win_tiebreak(p, p) == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("a,b", itertools.product(SERVE, SERVE))
def test_set_chances_add_up_to_one(a, b):
    ha, hb = prob_hold_serve(a), prob_hold_serve(b)
    first = prob_win_set(ha, hb, prob_win_tiebreak(a, b))
    second = prob_win_set(hb, ha, prob_win_tiebreak(b, a))
    assert first + second == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("a,b", itertools.product(SERVE, SERVE))
def test_match_chances_add_up_to_one(a, b, best_of):
    total = prob_win_match(a, b, best_of) + prob_win_match(b, a, best_of)
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("best_of", [3, 5])
def test_a_match_between_equals_is_even(best_of):
    for p in SERVE:
        assert prob_win_match(p, p, best_of) == pytest.approx(0.5, abs=1e-12)


def test_the_better_server_is_favoured_and_more_so_over_five_sets():
    assert prob_win_match(0.66, 0.62, 3) > 0.5
    assert prob_win_match(0.66, 0.62, 5) > prob_win_match(0.66, 0.62, 3)
    assert prob_win_match(0.70, 0.62, 3) > prob_win_match(0.66, 0.62, 3)

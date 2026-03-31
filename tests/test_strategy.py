"""
Tests: Strategy Logic — Milestone thresholds, pick selection, opener detection, shift recalibration.
"""

import pytest
from strategy.milestone_logic import (
    get_threshold, get_phase, select_picks, should_use_streak_saver, Pick
)
from strategy.opener_detector import apply_opener_adjustment
from strategy.shift_recalibration import apply_shift_recalibration, SHIFT_BABIP_PENALTIES


# ── Milestone Thresholds ────────────────────────────────────────────────────────

class TestMilestoneThresholds:

    def test_phase_aggressive_threshold(self):
        assert get_threshold(0)  == 0.80
        assert get_threshold(5)  == 0.80
        assert get_threshold(10) == 0.80

    def test_phase_opportunistic_threshold(self):
        assert get_threshold(11) == 0.85
        assert get_threshold(25) == 0.85
        assert get_threshold(40) == 0.85

    def test_phase_ultra_conservative_threshold(self):
        assert get_threshold(41) == 0.92
        assert get_threshold(50) == 0.92
        assert get_threshold(57) == 0.92

    def test_phase_beyond_57(self):
        # Should still apply maximum caution
        assert get_threshold(60) == 0.92

    def test_phase_names(self):
        assert get_phase(0)  == "aggressive"
        assert get_phase(11) == "opportunistic"
        assert get_phase(41) == "ultra_conservative"


# ── Pick Selection ──────────────────────────────────────────────────────────────

def _make_candidates(probs):
    """Helper: build a sorted candidate list from a list of P(Hit) values."""
    return [
        {
            "batter_id": 1000 + i,
            "game_date": "2025-06-15",
            "p_hit": p,
            "home_team": "NYY",
            "away_team": "BOS",
            "stand": "R",
            "explanation": "",
        }
        for i, p in enumerate(sorted(probs, reverse=True))
    ]


class TestPickSelection:

    def test_no_picks_when_below_threshold(self):
        candidates = _make_candidates([0.75, 0.72])
        picks = select_picks(candidates, streak_len=15)
        # Threshold at streak=15 is 0.85, all candidates are below
        assert len(picks) == 0

    def test_picks_above_threshold_are_selected(self):
        candidates = _make_candidates([0.91, 0.88, 0.75])
        picks = select_picks(candidates, streak_len=15)
        assert len(picks) == 2
        assert all(p.p_hit >= 0.85 for p in picks)

    def test_ultra_conservative_limits_to_one_pick(self):
        candidates = _make_candidates([0.95, 0.93, 0.92])
        picks = select_picks(candidates, streak_len=45)
        assert len(picks) == 1

    def test_double_down_applied_to_top_pick_in_aggressive_phase(self):
        candidates = _make_candidates([0.92, 0.87])
        picks = select_picks(candidates, streak_len=5, double_down_budget=1)
        top_pick = next(p for p in picks if p.p_hit == max(pk.p_hit for pk in picks))
        assert top_pick.double_down is True

    def test_double_down_not_applied_when_budget_zero(self):
        candidates = _make_candidates([0.92])
        picks = select_picks(candidates, streak_len=5, double_down_budget=0)
        assert all(not p.double_down for p in picks)

    def test_pick_result_is_pick_dataclass(self):
        candidates = _make_candidates([0.85])
        picks = select_picks(candidates, streak_len=0)
        assert isinstance(picks[0], Pick)

    def test_pick_to_dict_has_required_keys(self):
        candidates = _make_candidates([0.90])
        picks = select_picks(candidates, streak_len=0)
        d = picks[0].to_dict()
        for key in ("batter_id", "game_date", "p_hit", "home_team", "away_team", "stand", "double_down"):
            assert key in d


# ── Streak Saver Logic ──────────────────────────────────────────────────────────

class TestStreakSaver:

    def test_no_saver_when_none_remaining(self):
        assert should_use_streak_saver(0.40, streak_len=25, savers_remaining=0) is False

    def test_no_saver_in_aggressive_phase(self):
        # Phase 1 (streak ≤ 10): never waste a saver
        assert should_use_streak_saver(0.45, streak_len=5, savers_remaining=2) is False

    def test_saver_recommended_in_opportunistic_phase_low_prob(self):
        assert should_use_streak_saver(0.40, streak_len=25, savers_remaining=1) is True

    def test_no_saver_when_prob_above_threshold(self):
        assert should_use_streak_saver(0.75, streak_len=25, savers_remaining=1) is False


# ── Opener Adjustment (unit-level, no DB) ──────────────────────────────────────

class TestOpenerAdjustmentNoDb:
    """Test the opener adjustment logic without DuckDB (mocked detection)."""

    def test_adjustment_not_applied_when_detection_fails(self):
        """If opener detection fails, candidates should pass through unchanged."""
        candidates = [
            {"batter_id": 100, "home_team": "COL", "away_team": "LAD",
             "p_hit": 0.82, "stand": "R", "explanation": ""},
        ]
        # Pass a non-existent date — detection will fail gracefully
        result = apply_opener_adjustment(candidates, "2000-01-01")
        # p_hit should be unchanged (opener detection silently fails)
        assert result[0]["p_hit"] == 0.82


# ── Shift Recalibration (unit-level, no DB) ────────────────────────────────────

class TestShiftRecalibrationNoDb:

    def test_rhe_batters_not_affected(self):
        """Right-handed batters should not have shift penalty applied."""
        candidates = [
            {"batter_id": 200, "stand": "R", "p_hit": 0.85,
             "home_team": "TEX", "explanation": ""},
        ]
        result = apply_shift_recalibration(candidates, "2000-01-01")
        assert result[0]["p_hit"] == 0.85

    def test_shift_penalties_are_negative_or_zero(self):
        for penalty in SHIFT_BABIP_PENALTIES.values():
            assert penalty <= 0.0

    def test_4man_shift_has_largest_penalty(self):
        assert SHIFT_BABIP_PENALTIES["4-man shift"] <= SHIFT_BABIP_PENALTIES["Shifted"]
        assert SHIFT_BABIP_PENALTIES["Shifted"]     <= SHIFT_BABIP_PENALTIES["Strategic"]
        assert SHIFT_BABIP_PENALTIES["Strategic"]   <= SHIFT_BABIP_PENALTIES["Standard"]

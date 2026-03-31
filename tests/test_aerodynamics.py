"""
Tests: Physics Engine — Aerodynamics, Flight Model, Humidor.
"""

import math
import pytest
from physics.aerodynamics import (
    calc_air_density, density_delta, temp_to_distance_boost, air_density_score
)
from physics.flight_model import (
    estimate_fly_ball_distance, distance_boost_vs_baseline, temp_sensitivity_ft_per_10f
)
from physics.humidor import cor_adjustment, cor_to_distance_effect, humidor_babip_adjustment


class TestAerodynamics:

    def test_sea_level_standard_density(self):
        """At 59°F, 0 ft, 1013.25 mb → ρ ≈ 1.225 kg/m³."""
        rho = calc_air_density(temp_f=59.0, altitude_ft=0.0, pressure_mb=1013.25)
        assert abs(rho - 1.225) < 0.005, f"Expected ~1.225, got {rho:.4f}"

    def test_density_decreases_with_altitude(self):
        """Coors Field (5200 ft) should have lower density than sea level."""
        rho_sea   = calc_air_density(72.0, 0.0,    1013.25)
        rho_coors = calc_air_density(72.0, 5200.0, 1013.25)
        assert rho_coors < rho_sea, "Air should be thinner at altitude"

    def test_density_decreases_with_temperature(self):
        """Warmer air is less dense (ideal gas law)."""
        rho_cold = calc_air_density(50.0, 0.0, 1013.25)
        rho_hot  = calc_air_density(95.0, 0.0, 1013.25)
        assert rho_hot < rho_cold, "Hot air should be less dense"

    def test_density_delta_sea_level_is_zero_approximately(self):
        delta = density_delta(59.0, 0.0, 1013.25)
        assert abs(delta) < 0.01, f"Sea level delta should be ~0, got {delta}"

    def test_distance_boost_positive_when_hot(self):
        boost = temp_to_distance_boost(72.0, 92.0, 0.0, 1013.25)
        assert boost > 0, "Hot day should boost fly-ball distance"

    def test_air_density_score_in_range(self):
        score = air_density_score(80.0, 5200.0, 1013.25)
        assert -1.0 <= score <= 1.0
        assert score > 0, "Coors Field should have positive hitter-friendly score"


class TestFlightModel:

    def test_ten_degree_boost_approximately_3_5_ft(self):
        """Validate the +10°F ≈ +3.5 ft calibration at sea level."""
        boost = temp_sensitivity_ft_per_10f(
            exit_velo_mph=100.0, launch_angle_deg=28.0,
            altitude_ft=0.0, pressure_mb=1013.25,
        )
        assert 2.0 < boost < 10.0, f"Expected ~3.5 ft, got {boost:.2f}"

    def test_fly_ball_distance_reasonable_range(self):
        """A 100 mph, 28° fly ball should travel ~380-420 ft."""
        d = estimate_fly_ball_distance(100.0, 28.0, 72.0, 0.0, 1013.25)
        assert 300 < d < 500, f"Distance {d:.1f} ft seems unreasonable"

    def test_coors_advantage_over_sea_level(self):
        """Coors Field should provide positive distance boost."""
        boost = distance_boost_vs_baseline(
            exit_velo_mph=103, launch_angle_deg=28,
            temp_f=85, altitude_ft=5200, pressure_mb=1013.25,
        )
        assert boost > 0, f"Coors should boost distance, got {boost:.2f}"

    def test_launch_angle_affects_distance(self):
        """Distance should vary with launch angle (peak near 25-35°)."""
        d_low  = estimate_fly_ball_distance(100.0, 10.0, 72.0, 0.0, 1013.25)
        d_peak = estimate_fly_ball_distance(100.0, 28.0, 72.0, 0.0, 1013.25)
        d_high = estimate_fly_ball_distance(100.0, 55.0, 72.0, 0.0, 1013.25)
        assert d_peak > d_low
        assert d_peak > d_high


class TestHumidor:

    def test_cor_in_range(self):
        for stadium in ["Coors Field", "Fenway Park", "Chase Field"]:
            cor = cor_adjustment(stadium, ambient_rh_pct=50.0)
            assert 0.490 <= cor <= 0.545, f"{stadium}: COR={cor} out of range"

    def test_coors_humidor_normalizes_cor(self):
        """Coors with humidor at 50% RH target should have a depressed COR vs dry."""
        cor_coors_humidor = cor_adjustment("Coors Field", ambient_rh_pct=50.0)
        cor_dry_stadium   = cor_adjustment("Fenway Park", ambient_rh_pct=30.0)
        # Humidor should reduce COR toward neutral (less bounce)
        assert cor_coors_humidor <= cor_dry_stadium

    def test_higher_humidity_lower_cor(self):
        """Higher relative humidity in a non-humidor stadium → lower COR."""
        cor_dry = cor_adjustment("Fenway Park", ambient_rh_pct=30.0)
        cor_wet = cor_adjustment("Fenway Park", ambient_rh_pct=80.0)
        assert cor_wet < cor_dry

    def test_babip_adjustment_in_range(self):
        adj = humidor_babip_adjustment("Coors Field", 50.0)
        assert 0.85 <= adj <= 1.15

    def test_cor_distance_effect_direction(self):
        """Higher COR → positive distance effect relative to COR_DRY baseline."""
        d = cor_to_distance_effect(0.540)
        assert d > 0
        d2 = cor_to_distance_effect(0.510)
        assert d2 < 0

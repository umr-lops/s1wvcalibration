"""Wind bias contextualisation ranges (ported from qualitycheck/wind_bias_contextualisation.py)."""

from __future__ import annotations

wind_plages = {"lowwind": (0, 4), "mediumwind": (4, 11), "highwind": (11, 15)}
wind_pref_directions = {
    "upwind": (-45, 45),
    "downwind": (135, 225),
    "right_crosswind": (45, 135),
    "left_crosswind": (225, 315),
}

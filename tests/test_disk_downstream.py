"""Tests for the disk transition downstream stages (panel/sensitivity/interaction)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import run_food_transition_disk_interaction as inter  # noqa: E402
import run_food_transition_disk_panel as panel  # noqa: E402
import run_food_transition_disk_sensitivity as sens  # noqa: E402
from optimize_food_transition_disk import DISK_SEARCH_PARAMS, load_settings  # noqa: E402

CONFIG = ROOT / "configs" / "long_vertical_transition_disk.json"
UNPROJECT = ROOT / "configs" / "long_vertical_transition_disk_unproject.json"


class DiskDownstreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = load_settings(CONFIG)

    def test_unproject_config_is_disk_and_unproject(self) -> None:
        s = load_settings(UNPROJECT)
        self.assertEqual(s.food_geometry, "disk")
        self.assertEqual(s.direct_decode, "unproject")

    def test_candidate_params_fills_blank_from_base(self) -> None:
        # A blank fixed-parameter cell (food_value) falls back to the config.
        row = {name: "" for name in DISK_SEARCH_PARAMS}
        row["food_site_count"] = "6"
        row["food_site_radius"] = "200"
        params = panel.candidate_params(row, self.base)
        self.assertEqual(params["food_value"], self.base.food_value)
        self.assertEqual(params["food_site_count"], 6)
        self.assertIsInstance(params["food_site_count"], int)
        self.assertEqual(params["food_site_radius"], 200.0)

    def test_select_from_trials_dedups_and_ranks_by_stable(self) -> None:
        def make(number, stable, radius):
            row = {name: "0" for name in DISK_SEARCH_PARAMS}
            row.update(
                number=str(number), state="COMPLETE", value=str(stable),
                stable_count=str(stable), food_site_count="7",
                food_site_radius=str(radius), food_site_capacity="7",
                food_value="1.0", vertical_comb_benefit="0.5",
                food_site_max_distance="3250", travel_cost_per_distance="2e-5",
                mutation_sd="0.11", transposition_mutation_correlation="0.5",
            )
            return row

        rows = [make(1, 3, 200), make(2, 9, 210), make(3, 9, 210), make(4, 5, 150)]
        import csv, io, tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            path = Path(fh.name)
        selected = panel.select_from_trials(path, 10, self.base)
        path.unlink()
        # Highest stable first; the two identical radius-210 rows collapse to one.
        self.assertEqual(selected[0][0], "trial_2")
        names = [n for n, _ in selected]
        self.assertEqual(len(names), 3)  # 4 rows, one duplicate combo removed

    def test_sensitivity_points_include_clamped_baseline(self) -> None:
        baseline = {name: getattr(self.base, name) for name in DISK_SEARCH_PARAMS}
        baseline["vertical_comb_benefit"] = 0.56
        baseline["food_site_count"] = 8
        points = sens.build_points(baseline)
        self.assertEqual(points[0][0], "baseline")
        self.assertTrue(points[0][4])  # is_baseline
        # Ecology in every point matches baseline except the varied parameter.
        for name, params, parameter, _value, is_baseline in points[1:]:
            for key in DISK_SEARCH_PARAMS:
                if key != parameter:
                    self.assertEqual(params[key], baseline[key])
            self.assertNotEqual(params[parameter], baseline[parameter])
        # Benefit ladder is clamped at the 0.60 search ceiling (no +0.04 -> 0.60 ok,
        # but nothing exceeds it).
        benefit_vals = [
            p[1]["vertical_comb_benefit"] for p in points if p[2] == "vertical_comb_benefit"
        ]
        self.assertTrue(all(v <= 0.60 for v in benefit_vals))

    def test_interaction_grid_fixes_ecology_and_counts_cells(self) -> None:
        baseline = {name: getattr(self.base, name) for name in DISK_SEARCH_PARAMS}
        baseline["food_site_radius"] = 210.0
        cells = inter.build_cells(baseline)
        expected = (
            len(inter.VERTICAL_COMB_BENEFIT_VALUES)
            * len(inter.MUTATION_SD_VALUES)
            * len(inter.TRANSPOSITION_MUTATION_CORRELATION_VALUES)
        )
        self.assertEqual(len(cells), expected)
        for _name, params, grid in cells:
            # Ecology parameters stay at baseline; only the grid params change.
            self.assertEqual(params["food_site_radius"], 210.0)
            self.assertEqual(params["food_site_count"], baseline["food_site_count"])
            self.assertEqual(params["vertical_comb_benefit"], grid["vertical_comb_benefit"])


if __name__ == "__main__":
    unittest.main()

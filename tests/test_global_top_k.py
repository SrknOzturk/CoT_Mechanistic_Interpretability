import unittest

from experiments.run_patchings import run_id, select_global_top_units


class GlobalTopKTests(unittest.TestCase):
    def setUp(self):
        # These are distinct winners from the existing per-POS selection.
        self.candidates = {
            (0, 0): {"score": 0.4, "label": "NN"},
            (1, 0): {"score": 0.9, "label": "VB"},
            (2, 0): {"score": 0.2, "label": "NN"},
        }

    def test_margin_takes_highest_scores(self):
        selected = select_global_top_units(self.candidates, 2, largest=True)
        self.assertEqual(list(selected), [(1, 0), (0, 0)])

    def test_jsd_takes_lowest_scores(self):
        selected = select_global_top_units(self.candidates, 2, largest=False)
        self.assertEqual(list(selected), [(2, 0), (0, 0)])

    def test_unset_keeps_existing_selection(self):
        self.assertIs(select_global_top_units(self.candidates, None, True), self.candidates)

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            select_global_top_units(self.candidates, 0, True)

    def test_capped_runs_have_distinct_result_names(self):
        old = run_id("model", "data", "normal", "template", "mlp")
        capped = run_id("model", "data", "normal", "template", "mlp", 2)
        self.assertEqual(capped, old + "__topk2")


if __name__ == "__main__":
    unittest.main()

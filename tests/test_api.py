"""End-to-end HTTP API tests against the real standard-library server."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import build_server


class ServerFixture:
    def __init__(self):
        self.server = build_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def post(self, payload, raw=False):
        data = payload if raw else json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/v1/invert",
            data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read())


BODY = {
    "target": 100_000,
    "tolerance": 500,
    "components": [
        {"id": "monomer-A", "mass": 18_013, "min": 0, "max": 20},
        {"id": "adduct-Na", "mass": 22_990, "min": 0, "max": 10},
        {"id": "monomer-B", "mass": 44_000, "min": 0, "max": 5},
    ],
}


class ApiTests(unittest.TestCase):
    def test_health(self):
        with ServerFixture() as fx:
            status, body = fx.get("/healthz")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")

    def test_optimal_inversion(self):
        with ServerFixture() as fx:
            status, body = fx.post(BODY)
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "optimal")
            self.assertTrue(body["within_tolerance"])
            self.assertGreaterEqual(len(body["explanations"]), 1)
            expl = body["explanations"][0]
            # Every figure must be an integer and recompute exactly.
            self.assertIsInstance(expl["total_mass"], int)
            self.assertIsInstance(expl["error"], int)
            self.assertEqual(
                expl["recomputed_total_mass"],
                sum(c["count"] * c["mass"] for c in expl["counts"]))
            self.assertEqual(expl["recomputed_total_mass"], expl["total_mass"])
            self.assertEqual(expl["error"], expl["total_mass"] - BODY["target"])
            self.assertLessEqual(expl["absolute_error"], BODY["tolerance"])
            self.assertIn("unique", body)
            self.assertIsInstance(body["num_optimal_explanations"], int)
            # component ids present in declared order
            self.assertEqual(set(body["component_order"]),
                             {"monomer-A", "adduct-Na", "monomer-B"})
            for c in expl["counts"]:
                self.assertGreaterEqual(c["count"], c["min"])
                self.assertLessEqual(c["count"], c["max"])

    def test_non_unique_reports_alternative(self):
        # masses 4 and 6, target 11, tolerance 1 -> distance 1 both sides,
        # several minimum-count explanations.
        payload = {
            "target": 11, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 4, "min": 0, "max": 6},
                {"id": "b", "mass": 6, "min": 0, "max": 6},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 200)
            self.assertFalse(body["unique"])
            self.assertGreaterEqual(body["num_optimal_explanations"], 2)
            self.assertIsNotNone(body["alternative_witness"])

    def test_unsatisfiable_returns_witnesses(self):
        payload = {
            "target": 50, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 7, "min": 1, "max": 4},
                {"id": "b", "mass": 13, "min": 1, "max": 4},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "unsatisfiable")
            self.assertFalse(body["within_tolerance"])
            self.assertIsNotNone(body["nearest_below"])
            self.assertIsNotNone(body["nearest_above"])
            self.assertLessEqual(body["nearest_below"]["total_mass"], 50)
            self.assertGreaterEqual(body["nearest_above"]["total_mass"], 50)

    def test_structured_validation_errors_are_locatable(self):
        payload = {
            "target": -3,
            "tolerance": "x",
            "components": [
                {"id": "a", "mass": 1.5, "min": 5, "max": 2},
                {"id": "a", "mass": 0},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            paths = {e["path"]: e["code"] for e in body["errors"]}
            self.assertIn("/target", paths)
            self.assertIn("/tolerance", paths)
            self.assertIn("/components/0/mass", paths)
            self.assertIn("/components/0/min", paths)
            self.assertIn("/components/1/mass", paths)
            self.assertIn("/components/1/id", paths)

    def test_wrong_component_count(self):
        payload = {"target": 10, "tolerance": 1,
                   "components": [{"id": "a", "mass": 2}]}
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            self.assertEqual(body["errors"][0]["path"], "/components")

    def test_invalid_json(self):
        with ServerFixture() as fx:
            status, body = fx.post(b"{not json", raw=True)
            self.assertEqual(status, 400)
            self.assertEqual(body["error"]["code"], "invalid_json")

    def test_integer_only_enforcement(self):
        payload = {
            "target": 100.25, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 4},
                {"id": "b", "mass": 6},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            self.assertIn("/target",
                          [e["path"] for e in body["errors"]])

    def test_bound_limit_one_million(self):
        payload = {
            "target": 10**12, "tolerance": 10**6,
            "components": [
                {"id": "a", "mass": 123_456_789, "max": 1_000_001},
                {"id": "b", "mass": 987_654_321},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            self.assertIn("/components/0/max",
                          [e["path"] for e in body["errors"]])


# --------------------------------------------------------------------------- #
# Labeled-peak audit
#
# masses 4 and 6, target 11, tolerance 1: the nearest reachable masses are
# 10 (below) and 12 (above), both at distance 1.  Minimum-count winners:
#   (a=0, b=2) attaining 12  -- the canonical (lexicographically first) recipe
#   (a=1, b=1) attaining 10  -- a count-distinct alternative
# --------------------------------------------------------------------------- #

LABEL_COMPS = [
    {"id": "a", "mass": 4, "min": 0, "max": 6},
    {"id": "b", "mass": 6, "min": 0, "max": 6},
]


class LabelAuditApiTests(unittest.TestCase):
    def test_distinguishable_labels_and_nearest_outside_candidate(self):
        # +4 per labeled 'a' particle: canonical recipe (0,2) -> labeled 12;
        # alternative (1,1) -> labeled 14, error 2 > supplemental tolerance 1.
        payload = {
            "target": 11, "tolerance": 1, "components": LABEL_COMPS,
            "label_audit": {
                "label_mass_increments": [4, 0],
                "supplemental_tolerance": 1,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertFalse(body["unique"])
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "distinguishable")
        self.assertIsNone(audit["witness"])
        self.assertEqual(audit["label_mass_increments"], [4, 0])
        self.assertEqual(audit["supplemental_tolerance"], 1)

        ref_counts = {c["id"]: c["count"] for c in audit["reference_counts"]}
        self.assertEqual(ref_counts, {"a": 0, "b": 2})
        self.assertEqual(audit["reference_labeled_total_mass"], 12)

        cand = audit["nearest_outside_candidate"]
        self.assertIsNotNone(cand)
        cand_counts = {c["id"]: c for c in cand["counts"]}
        self.assertEqual(cand_counts["a"]["count"], 1)
        self.assertEqual(cand_counts["b"]["count"], 1)
        self.assertEqual(cand["total_mass"], 10)          # original main peak
        self.assertEqual(cand["labeled_total_mass"], 14)
        self.assertEqual(cand["reference_labeled_total_mass"], 12)
        self.assertEqual(cand["labeled_error"], 2)
        self.assertEqual(cand["absolute_labeled_error"], 2)
        self.assertFalse(cand["within_supplemental_tolerance"])
        self.assertEqual(cand["distance_to_tolerance_boundary"], 1)
        self.assertEqual(cand["side"], "above")
        # Every labeled figure is exact integer arithmetic and reconciles.
        self.assertEqual(
            cand["recomputed_labeled_total_mass"],
            sum(c["count"] * c["labeled_mass"] for c in cand["counts"]))
        self.assertEqual(cand_counts["a"]["label_mass_increment"], 4)
        self.assertEqual(cand_counts["a"]["labeled_mass"], 8)
        self.assertEqual(cand_counts["a"]["labeled_mass_contribution"], 8)
        self.assertEqual(cand_counts["b"]["labeled_mass_contribution"], 6)

    def test_indistinguishable_returns_count_distinct_witness(self):
        # +2 per labeled 'a': alternative (1,1) -> labeled 12, exactly on the
        # reference labeled mass and inside a zero supplemental tolerance.
        payload = {
            "target": 11, "tolerance": 1, "components": LABEL_COMPS,
            "label_audit": {
                "label_mass_increments": [2, 0],
                "supplemental_tolerance": 0,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "indistinguishable")
        self.assertIsNone(audit["nearest_outside_candidate"])
        witness = audit["witness"]
        counts = {c["id"]: c["count"] for c in witness["counts"]}
        self.assertEqual(counts, {"a": 1, "b": 1})  # genuinely different
        self.assertEqual(witness["total_mass"], 10)
        self.assertEqual(witness["labeled_total_mass"], 12)
        self.assertEqual(witness["labeled_error"], 0)
        self.assertTrue(witness["within_supplemental_tolerance"])
        self.assertEqual(witness["distance_to_tolerance_boundary"], 0)

    def test_tolerance_boundary_is_inclusive(self):
        # +1 per 'a': alternative labeled mass 11 vs reference 12, error -1.
        base = {
            "target": 11, "tolerance": 1, "components": LABEL_COMPS,
            "label_audit": {"label_mass_increments": [1, 0]},
        }
        with ServerFixture() as fx:
            st_in, body_in = fx.post({**base,
                "label_audit": {**base["label_audit"],
                                "supplemental_tolerance": 1}})
            st_out, body_out = fx.post({**base,
                "label_audit": {**base["label_audit"],
                                "supplemental_tolerance": 0}})
        self.assertEqual(st_in, 200)
        self.assertEqual(body_in["label_audit"]["conclusion"],
                         "indistinguishable")
        self.assertEqual(body_in["label_audit"]["witness"]["side"], "below")
        self.assertEqual(st_out, 200)
        self.assertEqual(body_out["label_audit"]["conclusion"],
                         "distinguishable")
        self.assertEqual(
            body_out["label_audit"]
                     ["nearest_outside_candidate"]
                     ["distance_to_tolerance_boundary"], 1)

    def test_scans_every_optimal_explanation_equal_mass_winners(self):
        # Equal masses: all of (0,2), (1,1), (2,0) attain the exact target 10
        # with two particles. Label 'a' by +2; only (1,1) at error 2 is the
        # nearest outside candidate -- (2,0) at error 4 must not be picked.
        payload = {
            "target": 10, "tolerance": 0,
            "components": [
                {"id": "a", "mass": 5, "min": 0, "max": 5},
                {"id": "b", "mass": 5, "min": 0, "max": 5},
            ],
            "label_audit": {
                "label_mass_increments": [2, 0],
                "supplemental_tolerance": 1,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["num_optimal_explanations"], 3)
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "distinguishable")
        cand = audit["nearest_outside_candidate"]
        counts = {c["id"]: c["count"] for c in cand["counts"]}
        self.assertEqual(counts, {"a": 1, "b": 1})
        self.assertEqual(cand["absolute_labeled_error"], 2)
        self.assertEqual(cand["distance_to_tolerance_boundary"], 1)

    def test_increments_align_with_submitted_component_order(self):
        # Submit 'b' first; increments [0, 4] must label 'a' (canonical first)
        # by 4 -- the same distinguishable case as the canonical-order request.
        payload = {
            "target": 11, "tolerance": 1,
            "components": [
                {"id": "b", "mass": 6, "min": 0, "max": 6},
                {"id": "a", "mass": 4, "min": 0, "max": 6},
            ],
            "label_audit": {
                "label_mass_increments": [0, 4],
                "supplemental_tolerance": 1,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["component_order"], ["a", "b"])
        audit = body["label_audit"]
        self.assertEqual(audit["label_mass_increments"], [4, 0])
        self.assertEqual(audit["conclusion"], "distinguishable")

    def test_truncated_winner_set_is_not_claimed_distinguishable(self):
        payload = {
            "target": 10, "tolerance": 0, "max_explanations": 2,
            "components": [
                {"id": "a", "mass": 5, "min": 0, "max": 5},
                {"id": "b", "mass": 5, "min": 0, "max": 5},
            ],
            "label_audit": {
                "label_mass_increments": [4, 0],
                "supplemental_tolerance": 1,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["truncated"])
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "inconclusive_truncated")
        self.assertIsNone(audit["witness"])
        self.assertIsNone(audit["nearest_outside_candidate"])

        # Collecting all winners turns the same audit into a hard verdict.
        payload["max_explanations"] = 100
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertFalse(body["truncated"])
        self.assertEqual(body["label_audit"]["conclusion"],
                         "distinguishable")

    def test_unique_optimum_needs_no_audit(self):
        # 6*5 = 30 and 10*3 = 30, but the five-6 recipe uses 5 particles vs
        # 3 for 10s; the two-level optimum is unique.
        payload = {
            "target": 30, "tolerance": 0,
            "components": [
                {"id": "a", "mass": 6, "min": 0, "max": 10},
                {"id": "b", "mass": 10, "min": 0, "max": 10},
            ],
            "label_audit": {
                "label_mass_increments": [2, 0],
                "supplemental_tolerance": 1,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["unique"])
        self.assertEqual(body["label_audit"]["conclusion"],
                         "unique_no_alternative")

    def test_audit_on_unsatisfiable_is_skipped(self):
        payload = {
            "target": 50, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 7, "min": 1, "max": 4},
                {"id": "b", "mass": 13, "min": 1, "max": 4},
            ],
            "label_audit": {
                "label_mass_increments": [2, 0],
                "supplemental_tolerance": 1,
            },
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "unsatisfiable")
        self.assertEqual(body["label_audit"]["conclusion"],
                         "skipped_unsatisfiable")

    def test_audit_is_off_by_default_all_legacy_shapes_unchanged(self):
        with ServerFixture() as fx:
            status, body = fx.post(BODY)
        self.assertEqual(status, 200)
        self.assertNotIn("label_audit", body)

        ambiguous = {
            "target": 11, "tolerance": 1, "components": LABEL_COMPS,
        }
        with ServerFixture() as fx:
            st_amb, amb = fx.post(ambiguous)
            st_un, uns = fx.post({
                "target": 50, "tolerance": 1,
                "components": [
                    {"id": "a", "mass": 7, "min": 1, "max": 4},
                    {"id": "b", "mass": 13, "min": 1, "max": 4},
                ],
            })
        self.assertEqual(st_amb, 200)
        self.assertEqual(amb["status"], "optimal")
        self.assertNotIn("label_audit", amb)
        self.assertEqual(st_un, 200)
        self.assertEqual(uns["status"], "unsatisfiable")
        self.assertNotIn("label_audit", uns)

    def test_audit_validation_errors_are_field_locatable(self):
        cases = [
            ({"label_mass_increments": [0, 0],
              "supplemental_tolerance": 1},
             "/label_audit/label_mass_increments", "all_zero"),
            ({"label_mass_increments": [1, 2, 3],
              "supplemental_tolerance": 1},
             "/label_audit/label_mass_increments", "out_of_range"),
            ({"label_mass_increments": [-1, 0],
              "supplemental_tolerance": 1},
             "/label_audit/label_mass_increments/0", "out_of_range"),
            ({"label_mass_increments": [1.5, 0],
              "supplemental_tolerance": 1},
             "/label_audit/label_mass_increments/0", "invalid_integer"),
            ({"label_mass_increments": [1, 0],
              "supplemental_tolerance": -2},
             "/label_audit/supplemental_tolerance", "out_of_range"),
            ({"label_mass_increments": [1, 0]},
             "/label_audit/supplemental_tolerance", "invalid_type"),
            ({"supplemental_tolerance": 1},
             "/label_audit/label_mass_increments", "invalid_type"),
            ({"label_mass_increments": "x",
              "supplemental_tolerance": 1},
             "/label_audit/label_mass_increments", "invalid_type"),
        ]
        for section, path, code in cases:
            with self.subTest(path=path):
                payload = {
                    "target": 11, "tolerance": 1, "components": LABEL_COMPS,
                    "label_audit": section,
                }
                with ServerFixture() as fx:
                    status, body = fx.post(payload)
                self.assertEqual(status, 400)
                by_path = {e["path"]: e["code"] for e in body["errors"]}
                self.assertEqual(by_path.get(path), code)

    def test_label_audit_section_must_be_object(self):
        payload = {
            "target": 11, "tolerance": 1, "components": LABEL_COMPS,
            "label_audit": [1, 0],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["errors"][0]["path"], "/label_audit")


if __name__ == "__main__":
    unittest.main()

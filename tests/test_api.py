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
# Isotope-label supplementary-peak audit
#
# masses 4 and 6, target 11, tolerance 1: optimal masses 10 (below) and 12
# (above), minimum particle count 2.  The canonical (lexicographically
# smallest) winner is (a=0, b=2) at mass 12; the alternative winner is
# (a=1, b=1) at mass 10.
# --------------------------------------------------------------------------- #

LABEL_BODY = {
    "target": 11, "tolerance": 1,
    "components": [
        {"id": "a", "mass": 4, "min": 0, "max": 6},
        {"id": "b", "mass": 6, "min": 0, "max": 6},
    ],
}


class LabelAuditApiTests(unittest.TestCase):
    def _counts(self, doc):
        return {c["id"]: c for c in doc["counts"]}

    def test_distinguishable_returns_nearest_counterexample(self):
        # Label component 'a' by +1: canonical labeled mass L* = 12, the
        # alternative (1, 1) lands at 11 -> gap 1; tolerance 0 excludes it.
        payload = {**LABEL_BODY, "label_audit": {
            "label_increments": [1, 0], "supplementary_tolerance": 0}}
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "distinguishable")
        self.assertEqual(audit["reference_labeled_mass"], 12)
        self.assertIsNone(audit["indistinguishable_witness"])
        near = audit["nearest_counterexample"]
        self.assertEqual(near["labeled_total_mass"], 11)
        self.assertEqual(near["labeled_mass_absolute_error"], 1)
        self.assertEqual(near["margin_to_tolerance"], 1)
        counts = self._counts(near)
        self.assertEqual(counts["a"]["count"], 1)
        self.assertEqual(counts["b"]["count"], 1)
        self.assertEqual(counts["a"]["labeled_mass"], 5)
        # Every figure recomputes exactly from the counts.
        self.assertEqual(
            near["labeled_total_mass"],
            sum(c["count"] * c["labeled_mass"] for c in near["counts"]))

    def test_indistinguishable_returns_tolerance_witness(self):
        payload = {**LABEL_BODY, "label_audit": {
            "label_increments": [1, 0], "supplementary_tolerance": 1}}
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "indistinguishable")
        self.assertIsNone(audit["nearest_counterexample"])
        witness = audit["indistinguishable_witness"]
        self.assertTrue(witness["within_supplementary_tolerance"])
        self.assertLessEqual(witness["labeled_mass_absolute_error"], 1)
        self.assertEqual(witness["labeled_total_mass"], 11)
        counts = self._counts(witness)
        self.assertEqual(counts["a"]["count"], 1)
        self.assertEqual(counts["b"]["count"], 1)
        # The witness must be a genuinely different count vector that still
        # belongs to the two-level optimum (same particle count as canonical).
        self.assertEqual(witness["particle_count"],
                         audit["canonical_explanation"]["particle_count"])
        self.assertNotEqual(
            [c["count"] for c in witness["counts"]],
            [c["count"] for c in audit["canonical_explanation"]["counts"]])

    def test_increments_align_with_request_component_order(self):
        # Submit components in non-canonical order (b before a); increments
        # still refer to the request array position.
        payload = {
            "target": 11, "tolerance": 1,
            "components": [
                {"id": "b", "mass": 6, "min": 0, "max": 6},
                {"id": "a", "mass": 4, "min": 0, "max": 6},
            ],
            "label_audit": {"label_increments": [0, 1],
                            "supplementary_tolerance": 0},
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        audit = body["label_audit"]
        self.assertEqual(audit["conclusion"], "distinguishable")
        increments = {e["id"]: e["increment"]
                      for e in audit["label_increments"]}
        self.assertEqual(increments, {"a": 1, "b": 0})
        self.assertEqual(audit["reference_labeled_mass"], 12)

    def test_audit_off_keeps_original_response_shape(self):
        # Without label_audit the responses must be byte-for-byte shaped as
        # before the feature: no audit block on success, ambiguity or the
        # unsatisfiable response.
        with ServerFixture() as fx:
            status, body = fx.post(LABEL_BODY)
            self.assertEqual(status, 200)
            self.assertNotIn("label_audit", body)

            status, body = fx.post(BODY)
            self.assertEqual(status, 200)
            self.assertNotIn("label_audit", body)

            unsat = {
                "target": 50, "tolerance": 1,
                "components": [
                    {"id": "a", "mass": 7, "min": 1, "max": 4},
                    {"id": "b", "mass": 13, "min": 1, "max": 4},
                ],
            }
            status, body = fx.post(unsat)
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "unsatisfiable")
            self.assertNotIn("label_audit", body)

    def test_unique_optimum_reports_unambiguous(self):
        payload = {**BODY, "label_audit": {
            "label_increments": [1, 0, 0], "supplementary_tolerance": 10}}
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 200)
        self.assertTrue(body["unique"])
        self.assertEqual(body["label_audit"]["conclusion"], "unambiguous")
        self.assertEqual(body["label_audit"]["reference_labeled_mass"],
                         body["label_audit"]["canonical_explanation"]
                         ["labeled_total_mass"])

    def test_audit_on_unsatisfiable_is_rejected(self):
        payload = {
            "target": 50, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 7, "min": 1, "max": 4},
                {"id": "b", "mass": 13, "min": 1, "max": 4},
            ],
            "label_audit": {"label_increments": [1, 0],
                            "supplementary_tolerance": 1},
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"]["code"], "label_audit_not_applicable")

    def test_label_audit_validation_errors_are_locatable(self):
        cases = [
            ({"label_increments": [1], "supplementary_tolerance": 1},
             "/label_audit/label_increments"),
            ({"label_increments": [-1, 0], "supplementary_tolerance": 1},
             "/label_audit/label_increments/0"),
            ({"label_increments": [0, 0], "supplementary_tolerance": 1},
             "/label_audit/label_increments"),
            ({"label_increments": [1, 0], "supplementary_tolerance": -1},
             "/label_audit/supplementary_tolerance"),
            ({"supplementary_tolerance": 1},
             "/label_audit/label_increments"),
        ]
        with ServerFixture() as fx:
            for audit, expected_path in cases:
                status, body = fx.post({**LABEL_BODY, "label_audit": audit})
                self.assertEqual(status, 400, audit)
                paths = [e["path"] for e in body["errors"]]
                self.assertIn(expected_path, paths)

    def test_label_audit_must_be_object(self):
        payload = {**LABEL_BODY, "label_audit": [1, 0]}
        with ServerFixture() as fx:
            status, body = fx.post(payload)
        self.assertEqual(status, 400)
        self.assertEqual(body["errors"][0]["path"], "/label_audit")


if __name__ == "__main__":
    unittest.main()

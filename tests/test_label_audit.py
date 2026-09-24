"""Differential tests for the optional labeled-peak audit.

For random small instances the full two-level optimum (nearest masses and
minimum-particle winners) and the audit verdict are recomputed by exhaustive
enumeration of the counting box in *canonical* (component-id) coordinates; the
service-layer audit must agree on

* the canonical reference vector (lexicographically smallest winner) and its
  labeled mass,
* the ``distinguishable`` / ``indistinguishable`` verdict,
* the witness distance or the nearest-outside candidate distance, and
* the item-by-item alignment of the increment array: requests submit
  components in a shuffled order, so the service must remap increments into
  canonical coordinates.
"""

from __future__ import annotations

import itertools
import random
import unittest

from app.service import invert


def brute_audit(masses, los, his, T, tol, inc, sup_tol):
    """Reference audit by full enumeration in canonical component order."""
    reachable: dict[int, list[tuple[int, ...]]] = {}
    ranges = [range(lo, hi + 1) for lo, hi in zip(los, his)]
    for vec in itertools.product(*ranges):
        mass = sum(c * m for c, m in zip(vec, masses))
        reachable.setdefault(mass, []).append(vec)

    best_dist = min(abs(m - T) for m in reachable)
    if best_dist > tol:
        return {"status": "unsatisfiable"}

    opt_masses = {m for m in reachable if abs(m - T) == best_dist}
    best_pc = min(sum(v) for m in opt_masses for v in reachable[m])
    winners = sorted(v for m in opt_masses for v in reachable[m]
                     if sum(v) == best_pc)
    if len(winners) == 1:
        ref = winners[0]
        return {"status": "unique", "ref": ref,
                "ref_lab": sum(c * (m + d)
                               for c, m, d in zip(ref, masses, inc))}

    ref = winners[0]
    ref_lab = sum(c * (m + d) for c, m, d in zip(ref, masses, inc))

    def lab_of(v):
        return sum(c * (m + d) for c, m, d in zip(v, masses, inc))

    inside_dists = [abs(lab_of(v) - ref_lab) for v in winners[1:]
                    if abs(lab_of(v) - ref_lab) <= sup_tol]
    if inside_dists:
        # Any tied witness is valid; only the minimum distance is forced.
        return {"status": "indistinguishable", "ref": ref,
                "ref_lab": ref_lab, "min_inside": min(inside_dists)}
    outside_dists = [abs(lab_of(v) - ref_lab) for v in winners[1:]]
    return {"status": "distinguishable", "ref": ref, "ref_lab": ref_lab,
            "nearest_distance": min(outside_dists)}


class LabelAuditDifferentialTests(unittest.TestCase):
    def test_random_instances(self):
        rng = random.Random(20260924)
        tried = non_unique = 0
        while tried < 1500:
            k = rng.randint(2, 4)
            ids = [f"c{i}" for i in range(k)]
            masses = [rng.randint(1, 25) for _ in range(k)]
            los = [rng.randint(0, 2) for _ in range(k)]
            his = [lo + rng.randint(0, 4) for lo in los]
            T = rng.randint(0, sum(m * h for m, h in zip(masses, his)) + 3)
            tol = rng.choice([0, 0, 1, 2, 4])
            inc = [rng.randint(0, 6) for _ in range(k)]
            if not any(inc):
                inc[rng.randrange(k)] = 1
            sup_tol = rng.choice([0, 0, 1, 2, 3, 8])

            # Reference truth in canonical id order.
            canon = sorted(range(k), key=lambda i: ids[i])
            c_ids = [ids[i] for i in canon]
            c_masses = [masses[i] for i in canon]
            c_los = [los[i] for i in canon]
            c_his = [his[i] for i in canon]
            c_inc = [inc[i] for i in canon]
            expected = brute_audit(c_masses, c_los, c_his, T, tol,
                                   c_inc, sup_tol)

            # Submit the same instance in a shuffled order; increments are
            # aligned item-by-item with the *submitted* components array.
            submit = list(range(k))
            rng.shuffle(submit)
            body = {
                "target": T, "tolerance": tol,
                "components": [
                    {"id": ids[i], "mass": masses[i],
                     "min": los[i], "max": his[i]}
                    for i in submit],
                "label_audit": {
                    "label_mass_increments": [inc[i] for i in submit],
                    "supplemental_tolerance": sup_tol,
                },
                "max_explanations": 10_000,
            }
            status, out = invert(body)
            self.assertEqual(status, 200)
            tried += 1
            audit = out["label_audit"]

            if expected["status"] == "unsatisfiable":
                self.assertEqual(out["status"], "unsatisfiable")
                self.assertEqual(audit["conclusion"], "skipped_unsatisfiable")
                continue
            self.assertEqual(out["status"], "optimal")
            self.assertFalse(out["truncated"])
            self.assertEqual(out["component_order"], c_ids)
            # The service must have remapped increments into canonical order.
            self.assertEqual(audit["label_mass_increments"], c_inc)

            def canonical_counts(counts):
                by_id = {c["id"]: c["count"] for c in counts}
                return tuple(by_id[cid] for cid in c_ids)

            self.assertEqual(canonical_counts(audit["reference_counts"]),
                             expected["ref"])
            self.assertEqual(audit["reference_labeled_total_mass"],
                             expected["ref_lab"])

            if expected["status"] == "unique":
                self.assertEqual(audit["conclusion"], "unique_no_alternative")
                self.assertTrue(out["unique"])
                continue
            non_unique += 1

            if expected["status"] == "indistinguishable":
                self.assertEqual(audit["conclusion"], "indistinguishable")
                w = audit["witness"]
                self.assertIsNotNone(w)
                # The witness must itself be a count-distinct winner and the
                # documented figures must reconcile exactly.
                self.assertNotEqual(canonical_counts(w["counts"]),
                                    expected["ref"])
                w_lab = sum(
                    cc["count"] * (cc["mass"] + cc["label_mass_increment"])
                    for cc in w["counts"])
                self.assertEqual(w["labeled_total_mass"], w_lab)
                self.assertEqual(w["recomputed_labeled_total_mass"], w_lab)
                self.assertLessEqual(
                    abs(w_lab - expected["ref_lab"]), sup_tol)
                self.assertEqual(w["absolute_labeled_error"],
                                 abs(w_lab - expected["ref_lab"]))
                self.assertTrue(w["within_supplemental_tolerance"])
            else:
                self.assertEqual(audit["conclusion"], "distinguishable")
                cand = audit["nearest_outside_candidate"]
                self.assertIsNotNone(cand)
                self.assertNotEqual(canonical_counts(cand["counts"]),
                                    expected["ref"])
                self.assertGreater(cand["absolute_labeled_error"], sup_tol)
                self.assertEqual(cand["absolute_labeled_error"],
                                 expected["nearest_distance"])
                self.assertEqual(
                    cand["distance_to_tolerance_boundary"],
                    cand["absolute_labeled_error"] - sup_tol)
                self.assertEqual(
                    cand["recomputed_labeled_total_mass"],
                    cand["labeled_total_mass"])
        # Sanity guard: the suite must actually exercise both verdicts.
        self.assertGreater(non_unique, 50)


if __name__ == "__main__":
    unittest.main()

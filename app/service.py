"""Application service: turn validated requests into response documents."""

from __future__ import annotations

import os
from typing import Any

from .errors import RequestError, ValidationError
from .solver import BudgetExceeded, Component, Solver
from .validation import MAX_BOUND, parse_request

DEFAULT_MAX_EXPLANATIONS = 100
HARD_MAX_EXPLANATIONS = 10_000


def _node_budget() -> int:
    raw = os.environ.get("SOLVER_NODE_BUDGET", "5000000")
    try:
        val = int(raw)
        return max(1_000, val)
    except ValueError:
        return 5_000_000


def _counts_doc(order: list[str], meta: dict[str, dict[str, int]],
                vec: list[int] | tuple[int, ...]) -> list[dict[str, int]]:
    return [
        {
            "id": cid,
            "mass": meta[cid]["mass"],
            "min": meta[cid]["min"],
            "max": meta[cid]["max"],
            "count": vec[pos],
            "mass_contribution": vec[pos] * meta[cid]["mass"],
        }
        for pos, cid in enumerate(order)
    ]


def _explanation(order: list[str], meta: dict[str, dict[str, int]],
                 mass: int, particle_count: int, target: int,
                 vec: tuple[int, ...]) -> dict[str, Any]:
    counts = _counts_doc(order, meta, vec)
    recomputed = sum(c["mass_contribution"] for c in counts)
    return {
        "total_mass": mass,
        "error": mass - target,                 # signed, in micro-daltons
        "absolute_error": abs(mass - target),
        "particle_count": particle_count,
        "counts": counts,
        "recomputed_total_mass": recomputed,    # equals total_mass exactly
    }


def _witness_doc(order: list[str], meta: dict[str, dict[str, int]],
                 witness: dict[str, Any] | None,
                 target: int) -> dict[str, Any] | None:
    if witness is None:
        return None
    return {
        "total_mass": witness["total_mass"],
        "error": witness["error"],
        "absolute_error": witness["absolute_error"],
        "particle_count": witness["particle_count"],
        "counts": _counts_doc(order, meta, witness["vector"]),
        "recomputed_total_mass": witness["total_mass"],
        "side": "below" if witness["error"] <= 0 else "above",
    }


def _labeled_counts_doc(order: list[str], meta: dict[str, dict[str, int]],
                        increments: list[int],
                        vec: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "id": cid,
            "mass": meta[cid]["mass"],
            "label_mass_increment": increments[pos],
            "labeled_mass": meta[cid]["mass"] + increments[pos],
            "min": meta[cid]["min"],
            "max": meta[cid]["max"],
            "count": vec[pos],
            "mass_contribution": vec[pos] * meta[cid]["mass"],
            "labeled_mass_contribution":
                vec[pos] * (meta[cid]["mass"] + increments[pos]),
        }
        for pos, cid in enumerate(order)
    ]


def _labeled_candidate_doc(order: list[str], meta: dict[str, dict[str, int]],
                           increments: list[int], vec: tuple[int, ...],
                           mass: int, reference_labeled: int,
                           sup_tol: int) -> dict[str, Any]:
    counts = _labeled_counts_doc(order, meta, increments, vec)
    labeled = sum(c["labeled_mass_contribution"] for c in counts)
    error = labeled - reference_labeled
    return {
        "total_mass": mass,
        "particle_count": sum(vec),
        "counts": counts,
        "labeled_total_mass": labeled,
        "reference_labeled_total_mass": reference_labeled,
        "labeled_error": error,                  # signed, exact
        "absolute_labeled_error": abs(error),
        "supplemental_tolerance": sup_tol,
        "within_supplemental_tolerance": abs(error) <= sup_tol,
        "distance_to_tolerance_boundary": abs(error) - sup_tol,
        "side": "below" if error <= 0 else "above",
        "recomputed_labeled_total_mass": labeled,
    }


def _label_audit_section(order: list[str], meta: dict[str, dict[str, int]],
                         increments_canonical: list[int], sup_tol: int,
                         result: dict[str, Any], optimal: bool) -> dict[str, Any]:
    """Audit other two-level-optimal explanations against the labeled peak.

    The canonical (lexicographically smallest) winning count vector fixes the
    reference labeled mass.  Every *other* optimal vector is an alternative
    explanation.  All arithmetic is exact integer arithmetic.
    """
    base = {
        "reference_component_order": order,
        "label_mass_increments": increments_canonical,
        "supplemental_tolerance": sup_tol,
        "reference_count_index": 0,
    }

    if not optimal:
        return {"conclusion": "skipped_unsatisfiable",
                "message": "no within-tolerance optimal set exists; the main "
                           "peak inversion was unsatisfiable",
                "witness": None, "nearest_outside_candidate": None, **base}

    canonical_vec = result["vectors"][0]
    reference_labeled = sum(
        canonical_vec[pos] * (meta[cid]["mass"] + increments_canonical[pos])
        for pos, cid in enumerate(order))
    base["reference_labeled_total_mass"] = reference_labeled
    base["reference_counts"] = _labeled_counts_doc(
        order, meta, increments_canonical, canonical_vec)

    if result["unique"]:
        return {"conclusion": "unique_no_alternative",
                "message": "the two-level optimum already has a unique count "
                           "explanation; no alternative needs excluding",
                "witness": None, "nearest_outside_candidate": None, **base}

    # Re-derive the attained mass per distinct optimal vector (vectors are in
    # canonical sorted order); the first one is the canonical reference.
    alternatives: list[tuple[tuple[int, ...], int]] = []
    for vec in result["vectors"][1:]:
        mass = sum(vec[pos] * meta[cid]["mass"]
                   for pos, cid in enumerate(order))
        alternatives.append((vec, mass))

    inside: list[tuple[int, tuple[int, ...], int]] = []
    outside: list[tuple[int, tuple[int, ...], int]] = []
    for vec, mass in alternatives:
        doc = _labeled_candidate_doc(order, meta, increments_canonical, vec,
                                     mass, reference_labeled, sup_tol)
        bucket = inside if doc["within_supplemental_tolerance"] else outside
        bucket.append((doc["absolute_labeled_error"], vec, mass))

    if inside:
        # A count-distinct vector whose labeled mass still lands in the
        # supplemental window: labeling cannot exclude the noncanonical
        # recipe.  Deterministic choice: smallest distance, then canonical
        # vector order.
        _, vec, mass = min(inside, key=lambda t: (t[0], t[1]))
        return {"conclusion": "indistinguishable",
                "message": "a count-distinct optimal explanation still lands "
                           "within the supplemental peak tolerance after "
                           "labeling",
                "witness": _labeled_candidate_doc(
                    order, meta, increments_canonical, vec, mass,
                    reference_labeled, sup_tol),
                "nearest_outside_candidate": None, **base}

    # All *collected* alternatives are outside.  If the winner list was
    # truncated, an uncollected explanation could still hide inside the
    # window, so distinguishability must not be claimed.
    if result["truncated_list"]:
        return {"conclusion": "inconclusive_truncated",
                "message": ("every collected alternative lies outside the "
                            "supplemental tolerance, but the optimal "
                            "explanation list was truncated; raise "
                            "'max_explanations' for an exact verdict"),
                "witness": None, "nearest_outside_candidate": None, **base}

    _, vec, mass = min(outside, key=lambda t: (t[0], t[1]))
    return {"conclusion": "distinguishable",
            "message": "every count-distinct optimal explanation falls "
                       "outside the supplemental peak tolerance after "
                       "labeling",
            "witness": None,
            "nearest_outside_candidate": _labeled_candidate_doc(
                order, meta, increments_canonical, vec, mass,
                reference_labeled, sup_tol),
            **base}


def invert(body: Any) -> tuple[int, dict[str, Any]]:
    """Execute one inversion. Returns (http_status, response_body)."""
    parsed = parse_request(body)

    max_collect = DEFAULT_MAX_EXPLANATIONS
    if isinstance(body, dict) and "max_explanations" in body:
        raw = body.get("max_explanations")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RequestError([ValidationError(
                "invalid_type",
                "'max_explanations' must be a positive integer",
                "/max_explanations")])
        max_collect = min(raw, HARD_MAX_EXPLANATIONS)

    components = [
        Component(id=c["id"], mass=c["mass"], lo=c["min"], hi=c["max"])
        for c in parsed["components"]
    ]
    meta = {
        c.id: {"mass": c.mass, "min": c.lo, "max": c.hi}
        for c in components
    }

    solver = Solver(components, parsed["target"], parsed["tolerance"],
                    node_budget=_node_budget())
    try:
        result = solver.solve(max_collect)
    except BudgetExceeded as exc:
        return 422, {
            "error": {
                "code": "search_budget_exceeded",
                "message": (f"{exc}; narrow the bounds/target or raise "
                            "SOLVER_NODE_BUDGET"),
            }
        }

    order = result["component_order"]
    target = result["target"]

    # Map the submitted per-component increments into canonical (id) order.
    audit = parsed.get("label_audit")
    if audit is not None:
        inc_by_id = {
            c["id"]: audit["increments"][i]
            for i, c in enumerate(parsed["components"])
        }
        increments_canonical = [inc_by_id[cid] for cid in order]
        sup_tol = audit["tolerance"]
    else:
        increments_canonical = []
        sup_tol = 0

    common = {
        "target": target,
        "tolerance": result["tolerance"],
        "mass_unit": "micro_dalton",
        "count_bounds_hard_max": MAX_BOUND,
        "component_order": order,
        "nearest_below": _witness_doc(order, meta, result["nearest_below"], target),
        "nearest_above": _witness_doc(order, meta, result["nearest_above"], target),
    }

    if result["status"] == "unsatisfiable":
        body = {
            "status": "unsatisfiable",
            "within_tolerance": False,
            "best_absolute_error": result["best_distance"],
            "message": ("no reachable total mass lies within the requested "
                        "tolerance; the closest attainable witnesses below "
                        "and above the target are provided"),
            **common,
        }
        if audit is not None:
            body["label_audit"] = _label_audit_section(
                order, meta, increments_canonical, sup_tol, result, False)
        return 200, body

    # Winners from the solver all attain one of the optimal masses; recover
    # the exact attained mass per vector (vectors are in canonical id order).
    optimal_set = set(result["optimal_masses"])
    explanations: list[dict[str, Any]] = []
    for vec in result["vectors"]:
        mass = sum(vec[pos] * solver.m[pos] for pos in range(solver.n))
        assert mass in optimal_set
        explanations.append(_explanation(
            order, meta, mass, result["particle_count"], target, vec))

    # A distinct witness proving non-uniqueness of the two-level optimum.
    alternative = None
    if not result["unique"] and len(explanations) >= 2:
        alternative = explanations[1]

    body = {
        "status": "optimal",
        "within_tolerance": True,
        "best_absolute_error": result["best_distance"],
        "optimal_total_masses": result["optimal_masses"],
        "particle_count": result["particle_count"],
        "num_optimal_explanations": result["num_optimal_explanations"],
        "unique": result["unique"],
        "alternative_witness": alternative,
        "truncated": result["truncated_list"],
        "explanations": explanations,
        **common,
    }
    if audit is not None:
        body["label_audit"] = _label_audit_section(
            order, meta, increments_canonical, sup_tol, result, True)
    return 200, body

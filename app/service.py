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

    # Label increments arrive aligned with the request's ``components`` array;
    # the solver works in canonical (id-sorted) order, so re-index by id.
    audit_spec = parsed.get("label_audit")
    increments_by_id: dict[str, int] | None = None
    if audit_spec is not None:
        increments_by_id = {
            c["id"]: audit_spec["label_increments"][pos]
            for pos, c in enumerate(parsed["components"])
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
        if audit_spec is not None:
            # The two-level optimal main peak does not exist here, so there is
            # no canonical composition whose labeled mass can serve as the
            # audit reference.
            return 422, {
                "error": {
                    "code": "label_audit_not_applicable",
                    "message": ("no reachable total mass lies within the main "
                                "tolerance; the label audit requires a "
                                "two-level optimal main peak"),
                },
                **common,
            }
        return 200, {
            "status": "unsatisfiable",
            "within_tolerance": False,
            "best_absolute_error": result["best_distance"],
            "message": ("no reachable total mass lies within the requested "
                        "tolerance; the closest attainable witnesses below "
                        "and above the target are provided"),
            **common,
        }

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

    response: dict[str, Any] = {
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

    if audit_spec is not None:
        increments = [increments_by_id[cid] for cid in order]
        if result["unique"]:
            response["label_audit"] = _label_audit_summary(
                order, meta, increments,
                audit_spec["supplementary_tolerance"],
                conclusion="unambiguous",
                message=("the two-level optimum is already unique; the "
                         "supplementary peak has no alternative to rule out"),
                canonical=result["vectors"][0],
                other=None,
                other_key=None)
        else:
            try:
                audit = solver.label_audit(
                    result["winning_blocks"], result["vectors"],
                    increments, audit_spec["supplementary_tolerance"],
                    truncated=result["truncated_list"])
            except BudgetExceeded as exc:
                return 422, {
                    "error": {
                        "code": "search_budget_exceeded",
                        "message": (f"{exc}; narrow the bounds/target or raise "
                                    "SOLVER_NODE_BUDGET"),
                    },
                }
            if audit["conclusion"] == "distinguishable":
                other_key = "nearest_counterexample"
                other_vec = audit["nearest_vector"]
            else:
                other_key = "indistinguishable_witness"
                other_vec = audit["witness_vector"]
            response["label_audit"] = _label_audit_summary(
                order, meta, increments,
                audit_spec["supplementary_tolerance"],
                conclusion=audit["conclusion"],
                message=None,
                canonical=audit["canonical_vector"],
                other=other_vec,
                other_key=other_key,
                audit=audit)

    return 200, response


def _label_counts(order: list[str], meta: dict[str, dict[str, int]],
                  increments: list[int], vec: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "id": cid,
            "mass": meta[cid]["mass"],
            "min": meta[cid]["min"],
            "max": meta[cid]["max"],
            "count": vec[pos],
            "mass_contribution": vec[pos] * meta[cid]["mass"],
            "label_increment": increments[pos],
            "labeled_mass": meta[cid]["mass"] + increments[pos],
            "labeled_contribution": vec[pos] * (meta[cid]["mass"] + increments[pos]),
        }
        for pos, cid in enumerate(order)
    ]


def _labeled_explanation(order: list[str], meta: dict[str, dict[str, int]],
                         increments: list[int],
                         vec: tuple[int, ...]) -> dict[str, Any]:
    counts = _label_counts(order, meta, increments, vec)
    total = sum(c["mass_contribution"] for c in counts)
    labeled = sum(c["labeled_contribution"] for c in counts)
    return {
        "particle_count": sum(vec),
        "total_mass": total,
        "labeled_total_mass": labeled,
        "label_shift": labeled - total,
        "counts": counts,
    }


def _label_audit_summary(order: list[str], meta: dict[str, dict[str, int]],
                         increments: list[int], supplementary_tolerance: int,
                         *, conclusion: str, message: str | None,
                         canonical: tuple[int, ...],
                         other: tuple[int, ...] | None,
                         other_key: str | None,
                         audit: dict[str, Any] | None = None) -> dict[str, Any]:
    canon_doc = _labeled_explanation(order, meta, increments, canonical)
    summary: dict[str, Any] = {
        "conclusion": conclusion,
        "supplementary_tolerance": supplementary_tolerance,
        "label_increments": [
            {"id": cid, "increment": increments[pos]}
            for pos, cid in enumerate(order)
        ],
        "reference_labeled_mass": canon_doc["labeled_total_mass"],
        "canonical_explanation": canon_doc,
    }
    if message is not None:
        summary["message"] = message
    if other is not None and audit is not None and other_key is not None:
        other_doc = _labeled_explanation(order, meta, increments, other)
        other_doc["labeled_mass_error"] = (
            other_doc["labeled_total_mass"] - canon_doc["labeled_total_mass"])
        other_doc["labeled_mass_absolute_error"] = abs(
            other_doc["labeled_mass_error"])
        if other_key == "nearest_counterexample":
            other_doc["margin_to_tolerance"] = audit["margin_to_tolerance"]
            summary["nearest_counterexample"] = other_doc
            summary["indistinguishable_witness"] = None
        else:
            other_doc["within_supplementary_tolerance"] = True
            summary["indistinguishable_witness"] = other_doc
            summary["nearest_counterexample"] = None
    return summary

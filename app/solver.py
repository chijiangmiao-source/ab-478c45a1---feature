"""Exact branch-and-bound inversion over a bounded counting box.

Model (all quantities are arbitrary-precision Python ``int``)::

    M = sum_i x_i * m_i,        lo_i <= x_i <= hi_i,   m_i > 0

The service must never *expand* a counting interval.  Everything here walks the
box with recursive branch-and-bound; no interval is materialized, no dynamic
program is laid out over any count interval, and no general optimization
solver is used.

Two-level optimization
----------------------
1. **Quality**  minimize |M - target| over all reachable masses in the box.
   Exact nearest reachable masses on either side of the target are found with a
   DFS pruned by suffix mass windows, suffix-gcd congruence feasibility and an
   incumbent seeded by greedy filling/trimming.
2. **Particle count**  among *every* count vector attaining an optimal mass,
   minimize sum x_i.  A memoized suffix recursion over ``(position, residual)``
   computes the exact minimum and the exact (arbitrary precision) number of
   attaining vectors, which decides uniqueness; concrete witnesses are
   enumerated on demand in lexicographically canonical order.
3. **Label audit (optional).**  When the analyst submits per-component
   isotope-label mass increments plus a supplementary-peak tolerance, the
   minimum-count DAG is reused: a second suffix memo records the min/max
   reachable *labeled* mass at every DAG state, and pruned walks decide
   exactly whether any other optimal explanation survives the supplementary
   peak.

All phases only ever use integer arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Optional

# The unlimited-coin residue oracle costs O(n * m_min) time and memory; only
# build it when the smallest mass is at most this many micro-daltons.
ORACLE_MAX_MODULUS = 2_000_000
# Gates scan residues directly while the incumbent gap window is this small;
# wider windows fall back to block-indexed range minima.
ORACLE_SCAN_LIMIT = 512
_ORACLE_BLOCK = 256
_INF = 10**30


class BudgetExceeded(Exception):
    """Raised when a search visits more nodes than the configured budget."""


@dataclass(frozen=True)
class Component:
    id: str
    mass: int
    lo: int
    hi: int


_INFEASIBLE = object()


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for b > 0, exact for possibly-negative a."""
    return -((-a) // b)


class Solver:
    def __init__(self, components: list[Component], target: int, tolerance: int,
                 node_budget: int = 5_000_000):
        # Canonical coordinate order: by component identifier.
        comps = sorted(components, key=lambda c: c.id)
        self.ids = [c.id for c in comps]
        self.m = [c.mass for c in comps]
        self.lo = [c.lo for c in comps]
        self.hi = [c.hi for c in comps]
        self.n = len(comps)
        self.T = target
        self.tol = tolerance
        self.node_budget = node_budget
        self.nodes = 0

        self.base = sum(self.lo[i] * self.m[i] for i in range(self.n))
        self.top = sum(self.hi[i] * self.m[i] for i in range(self.n))
        self.cap = [self.hi[i] - self.lo[i] for i in range(self.n)]

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.node_budget:
            raise BudgetExceeded(f"search visited {self.nodes} nodes")

    # ------------------------------------------------------------------ #
    # Phase 1: nearest reachable mass on each side of the target
    # ------------------------------------------------------------------ #

    def _greedy_below(self) -> Optional[tuple[int, tuple[int, ...]]]:
        """Largest-ish reachable mass <= T via greedy filling (seed only)."""
        if self.base > self.T:
            return None
        x = self.lo[:]
        total = self.base
        for j in sorted(range(self.n), key=lambda i: -self.m[i]):
            inc = min(self.cap[j], (self.T - total) // self.m[j])
            if inc:
                x[j] += inc
                total += inc * self.m[j]
        return total, tuple(x)

    def _greedy_above(self) -> Optional[tuple[int, tuple[int, ...]]]:
        """Smallest-ish reachable mass >= T via greedy trimming (seed only)."""
        if self.top < self.T:
            return None
        x = self.hi[:]
        total = self.top
        for j in sorted(range(self.n), key=lambda i: -self.m[i]):
            dec = min(x[j] - self.lo[j], (total - self.T) // self.m[j])
            if dec:
                x[j] -= dec
                total -= dec * self.m[j]
        return total, tuple(x)

    # ------------------------------------------------------------------ #
    # Phase 1: nearest reachable mass on each side of the target
    # ------------------------------------------------------------------ #

    def _build_oracle(self):
        """Residue shortest-path oracle over *unbounded* coins.

        Chooses the smallest mass m0 as modulus and computes, for every
        residue r, the minimum representable mass ``o[r]`` congruent to
        r (mod m0) over an *unbounded* relaxation of every coin (count caps
        ignored).  Because m0 itself is a coin, the unbounded representable
        values of residue r are exactly ``o[r] + z*m0`` for z >= 0.

        Per added coin the update is a min-plus closure on residue cycles;
        each cycle is covered by one forward and one backward sweep, giving
        O(m0) work per coin -- independent of all count bounds, so no
        counting interval is ever expanded.

        Returns ``(m0_index, m0, dist)`` or ``None`` when m0 is too large.
        """
        k = min(range(self.n), key=lambda i: self.m[i])
        m0 = self.m[k]
        if m0 > ORACLE_MAX_MODULUS:
            return None
        dist = [_INF] * m0
        dist[0] = 0

        for w in self.m:
            r = w % m0
            if r == 0:
                # Residue-0 coin: only useful at value 0 (other multiples are
                # strictly heavier residue-0 sums).
                continue
            g = gcd(r, m0)
            length = m0 // g
            for s in range(g):
                # Cycle v_t = (s + t*r) mod m0, t = 0..length-1.
                orig = [0] * length
                v = s
                for t in range(length):
                    orig[t] = dist[v]
                    v = (v + r) % m0
                wr = t  # silence linters; recomputed below implicitly
                del wr
                # Non-wrapping predecessors j <= t:
                #   best[t] = min_j (orig[j] + (t-j)*w)
                # Wrapping predecessors j > t:
                #   best[t] = min_j (orig[j] + (t + length - j)*w)
                best = [_INF] * length
                run = _INF
                for t in range(length):
                    cand = orig[t] - t * w
                    if cand < run:
                        run = cand
                    bt = run + t * w
                    if bt < best[t]:
                        best[t] = bt
                suf = _INF
                for t in range(length - 1, -1, -1):
                    bt = suf + (t + length) * w
                    if bt < best[t]:
                        best[t] = bt
                    cand = orig[t] - t * w
                    if cand < suf:
                        suf = cand
                v = s
                for t in range(length):
                    if best[t] < dist[v]:
                        dist[v] = best[t]
                    v = (v + r) % m0

        if dist[0] != 0:
            dist[0] = 0
        return k, m0, dist

    def _extreme(self, side: int, oracle=None) -> Optional[tuple[int, tuple[int, ...]]]:
        """Extreme reachable mass relative to the target.

        side = -1  ->  maximize M with M <= T (nearest reachable mass below)
        side = +1  ->  minimize M with M >= T (nearest reachable mass above)

        Returns ``(mass, count_vector)`` or ``None`` when no reachable mass
        exists on that side of the target.
        """
        n, m, T = self.n, self.m, self.T

        # Suffix information over *effective* counts y_i = x_i - lo_i.
        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + self.cap[i] * m[i]
            sg[i] = gcd(m[i], sg[i + 1])

        seed = self._greedy_below() if side == -1 else self._greedy_above()
        best: Optional[int] = seed[0] if seed is not None else None
        best_vec: Optional[tuple[int, ...]] = seed[1] if seed is not None else None
        if best == T:
            return best, best_vec  # type: ignore[return-value]

        om: int = oracle[1] if oracle else 0
        od: Optional[list[int]] = oracle[2] if oracle else None
        x = self.lo[:]

        def oracle_gate(lo_q: int, hi_q: int) -> bool:
            """O(1) exact necessary condition over the unbounded relaxation.

            Rejects only when no value of the required residue class can fall
            inside [lo_q, hi_q] even with caps removed -- hence rejection is
            sound for the true bounded suffix.
            """
            if od is None:
                return True
            d = od[lo_q % om]
            if d == _INF or d > hi_q:
                return False
            # Smallest representable value >= lo_q in this residue class.
            if d < lo_q:
                d += _ceil_div(lo_q - d, om) * om
            return d <= hi_q

        def suffix_gate(i: int, S: int) -> bool:
            """Prune node (i, S): suffix range, gcd congruence, incumbent."""
            nonlocal best, best_vec
            g = sg[i]
            if side == -1:
                hi_q = min(smax[i], T - S)
                if hi_q < 0:
                    return False  # S > T and all masses are positive
                q = hi_q if g == 0 else hi_q - (hi_q % g)
                if q < 0:
                    return False
                if best is not None and S + q <= best:
                    return False
                lo_q = max(0, best + 1 - S) if best is not None else 0
                if lo_q > hi_q:
                    return False
                return oracle_gate(lo_q, hi_q)
            lo_q = max(0, T - S)
            if lo_q > smax[i]:
                return False  # even a full suffix cannot reach T
            if g == 0:
                q = 0
            else:
                q = lo_q + ((-lo_q) % g)
            if q > smax[i]:
                return False
            if best is not None and S + q >= best:
                return False
            hi_q = min(smax[i], best - 1 - S) if best is not None else smax[i]
            if lo_q > hi_q:
                return False
            return oracle_gate(lo_q, hi_q)

        def dfs(i: int, S: int) -> None:
            nonlocal best, best_vec
            if best == T:
                return
            self._tick()
            if not suffix_gate(i, S):
                return
            if i == n:
                if side == -1 and S <= T and (best is None or S > best):
                    best, best_vec = S, tuple(x)
                elif side == 1 and S >= T and (best is None or S < best):
                    best, best_vec = S, tuple(x)
                return

            m_i = m[i]
            suffix_max = smax[i + 1]

            # Window of actual counts c for component i worth visiting.
            if side == -1:
                # S + (c - lo)*m_i <= T
                c_hi = min(self.hi[i], self.lo[i] + (T - S) // m_i)
                c_lo = self.lo[i]
                if best is not None:
                    # subtree must be able to beat incumbent even filled full:
                    # S + (c-lo)*m_i + suffix_max > best
                    c_lo = max(c_lo, self.lo[i]
                               + (best - S - suffix_max) // m_i + 1)
            else:
                # S + (c-lo)*m_i + suffix_max >= T
                need = T - S - suffix_max
                c_lo = max(self.lo[i], self.lo[i] + _ceil_div(need, m_i))
                c_hi = self.hi[i]
                if best is not None:
                    # subtree's emptiest mass must stay strictly below best:
                    # S + (c-lo)*m_i < best
                    c_hi = min(c_hi, self.lo[i] + (best - 1 - S) // m_i)

            if c_lo > c_hi:
                return

            # Start at the count whose raw total is closest to T so that the
            # incumbent tightens immediately; then alternate outward.
            ideal = self.lo[i] + (T - S) // m_i
            c0 = c_lo if ideal < c_lo else c_hi if ideal > c_hi else ideal

            def visit(c: int) -> None:
                x[i] = c
                dfs(i + 1, S + (c - self.lo[i]) * m_i)
                x[i] = self.lo[i]

            visit(c0)
            step = 1
            while best != T:
                down = c0 - step
                up = c0 + step
                if down < c_lo and up > c_hi:
                    break
                # For the below side probe smaller counts first (they cannot
                # overshoot); above side symmetrically probes larger counts.
                if side == -1:
                    if down >= c_lo:
                        visit(down)
                    if up <= c_hi:
                        visit(up)
                else:
                    if up <= c_hi:
                        visit(up)
                    if down >= c_lo:
                        visit(down)
                step += 1

        dfs(0, self.base)
        if best is None or best_vec is None:
            return None
        return best, best_vec

    # ------------------------------------------------------------------ #
    # Phase 2: minimum particle count at a fixed exact mass
    # ------------------------------------------------------------------ #

    def _build_min_count_dag(self, mass: int,
                            order: Optional[list[int]] = None) -> dict:
        """Build the memoized suffix DAG of minimum-count attaining vectors.

        Coordinates follow ``order`` (default: heaviest masses first, which
        pushes the residual onto few large coins and keeps explored counts
        small).  The DAG shares one ``solve(i, R)`` oracle:

        * ``memo[i][R] = (minimum suffix particle count, number of ways)``
        * an edge ``y`` out of ``(i, R)`` is optimal exactly when
          ``y + solve(i + 1, R - y*mm[i])[0] == memo[i][R][0]``.

        Root-to-leaf paths are precisely the minimum-count vectors attaining
        the requested mass; nothing here expands a count interval.
        """
        self.nodes = 0
        R0 = mass - self.base
        if R0 < 0:
            raise ValueError("mass below box minimum")

        if order is None:
            order = sorted(range(self.n), key=lambda i: -self.m[i])
        mm = [self.m[i] for i in order]
        cc = [self.cap[i] for i in order]
        n = self.n

        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + cc[i] * mm[i]
            sg[i] = gcd(mm[i], sg[i + 1])

        if R0 > smax[0] or R0 % sg[0] != 0:
            raise ValueError("mass not attainable")

        # memo[i][R] = (minimum suffix count, number of ways) or _INFEASIBLE.
        memo: list[dict[int, object]] = [dict() for _ in range(n + 1)]

        def solve(i: int, R: int) -> Optional[tuple[int, int]]:
            if R == 0:
                return (0, 1)
            if i == n or R > smax[i] or R % sg[i] != 0:
                return None
            cached = memo[i].get(R, _INFEASIBLE)
            if cached is not _INFEASIBLE:
                return cached  # type: ignore[return-value]
            self._tick()

            m_i, cap_i = mm[i], cc[i]
            g2 = sg[i + 1]

            # Residual R - y*m_i must lie inside [0, smax[i+1]].
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            if y_lo > y_hi:
                memo[i][R] = None
                return None

            cand = _congruence_candidates(y_lo, y_hi, m_i, R, g2)
            best_k: Optional[int] = None
            best_ways = 0
            if cand is not None:
                y, step = cand
                while y <= y_hi:
                    if best_k is not None and y > best_k:
                        break  # suffix counts are non-negative
                    sub = solve(i + 1, R - y * m_i)
                    if sub is not None:
                        k = y + sub[0]
                        if best_k is None or k < best_k:
                            best_k, best_ways = k, sub[1]
                        elif k == best_k:
                            best_ways += sub[1]
                    y += step

            result = None if best_k is None else (best_k, best_ways)
            memo[i][R] = result
            return result

        root = solve(0, R0)
        if root is None:
            raise ValueError("mass not attainable")

        return {
            "mass": mass,
            "R0": R0,
            "order": order,
            "mm": mm,
            "cc": cc,
            "smax": smax,
            "sg": sg,
            "memo": memo,
            "solve": solve,
            "root": root,
        }

    def _collect_minimal(self, dag: dict, max_collect: int) -> list[tuple[int, ...]]:
        """Up to ``max_collect`` minimum-count vectors, mapped to id order.

        Vectors are reached through optimal DAG edges only.
        """
        n = self.n
        mm, cc = dag["mm"], dag["cc"]
        order, smax, sg = dag["order"], dag["smax"], dag["sg"]
        memo, solve, R0 = dag["memo"], dag["solve"], dag["R0"]
        collected: list[tuple[int, ...]] = []

        def collect(i: int, R: int, prefix: list[int]) -> None:
            if len(collected) >= max_collect:
                return
            if R == 0:
                # Remaining DAG coordinates are pinned at their lower bound.
                eff = prefix + [0] * (n - len(prefix))
                vec = [0] * n
                for pos, idx in enumerate(order):
                    vec[idx] = self.lo[idx] + eff[pos]
                collected.append(tuple(vec))
                return
            entry = memo[i].get(R, _INFEASIBLE)
            if entry is None or entry is _INFEASIBLE:
                return
            target_k = entry[0]
            m_i, cap_i = mm[i], cc[i]
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            if cand is None:
                return
            y, step = cand
            while y <= y_hi and y <= target_k:
                sub = solve(i + 1, R - y * m_i)
                if sub is not None and y + sub[0] == target_k:
                    prefix.append(y)
                    collect(i + 1, R - y * m_i, prefix)
                    prefix.pop()
                y += step

        collect(0, R0, [])
        return collected

    def min_particles(self, mass: int, max_collect: int) -> dict:
        """Minimum-count vectors attaining the exact ``mass``.

        Returns a dict with the minimum particle count, the *exact* number of
        vectors attaining it (arbitrary precision), up to ``max_collect``
        witness vectors in canonical (id) order, and a truncation flag.
        """
        dag = self._build_min_count_dag(mass)
        min_extra, num_ways = dag["root"]
        if dag["R0"] == 0:
            return {
                "particle_count": sum(self.lo),
                "num_vectors": 1,
                "vectors": [tuple(self.lo)],
                "truncated": False,
            }
        collected = self._collect_minimal(dag, max_collect)
        return {
            "particle_count": sum(self.lo) + min_extra,
            "num_vectors": num_ways,
            "vectors": sorted(collected),
            "truncated": num_ways > len(collected),
        }

    # ------------------------------------------------------------------ #
    # Phase 3 (optional): isotope-label supplementary-peak audit
    # ------------------------------------------------------------------ #

    def _canonical_winner(self, winning_blocks: list[dict],
                          collected: list[tuple[int, ...]],
                          truncated: bool) -> tuple[int, ...]:
        """Lexicographically smallest second-level winner (canonical id order).

        A complete, already gathered winner list names the answer directly.
        When gathering was truncated the gathered prefix cannot be trusted to
        contain the global minimum, so a dedicated canonical-order DAG walks
        the single lexicographically smallest optimal path per optimal mass
        without expanding any count interval.
        """
        if collected and not truncated:
            return collected[0]
        best: Optional[tuple[int, ...]] = None
        for block in winning_blocks:
            dag = self._build_min_count_dag(block["mass"], list(range(self.n)))
            mm, cc = dag["mm"], dag["cc"]
            smax, sg, memo = dag["smax"], dag["sg"], dag["memo"]
            solve, R0 = dag["solve"], dag["R0"]
            chosen: list[int] = []
            R = R0
            for i in range(self.n):
                if R == 0:
                    # Remaining coordinates are pinned at their lower bound.
                    chosen.extend([0] * (self.n - i))
                    break
                target_k = memo[i][R][0]
                y_hi = min(cc[i], R // mm[i], target_k)
                y_lo = max(0, _ceil_div(R - smax[i + 1], mm[i]))
                cand = _congruence_candidates(y_lo, y_hi, mm[i], R, sg[i + 1])
                picked: Optional[int] = None
                if cand is not None:
                    y, step = cand
                    while y <= y_hi:
                        sub = solve(i + 1, R - y * mm[i])
                        if sub is not None and y + sub[0] == target_k:
                            picked = y
                            break
                        y += step
                if picked is None:  # impossible on a feasible DAG
                    raise ValueError("mass not attainable")
                chosen.append(picked)
                R -= picked * mm[i]
            vec = tuple(self.lo[i] + chosen[i] for i in range(self.n))
            if best is None or vec < best:
                best = vec
        assert best is not None
        return best

    def _label_extrema(self, dag: dict,
                       labeled_m: list[int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
        """Min/max reachable labeled suffix sum at every optimal DAG state.

        For state ``(i, R)`` only optimal min-count edges are considered, so
        leaves are exactly the block's minimum-count explanations.  Labeled
        suffix sums use effective counts ``y`` (the constant ``sum lo*lm``
        offset is added by the caller).  Returns a memo mapping

            (i, R) -> (min_Q, edge_to_min, max_Q, edge_to_max)

        States are the min-count DAG states only; like every other phase no
        counting interval is expanded.
        """
        order, mm, cc = dag["order"], dag["mm"], dag["cc"]
        smax, sg, minmemo, solve = (dag["smax"], dag["sg"],
                                    dag["memo"], dag["solve"])
        n = self.n
        lm = [labeled_m[idx] for idx in order]
        ext: dict[tuple[int, int], tuple[int, int, int, int]] = {}

        def run(i: int, R: int) -> tuple[int, int, int, int]:
            if R == 0:
                # Trailing components are pinned at their lower bound, i.e.
                # effective count 0 on every remaining edge.
                return (0, 0, 0, 0)
            cached = ext.get((i, R))
            if cached is not None:
                return cached
            self._tick()
            target_k = minmemo[i][R][0]
            m_i, cap_i = mm[i], cc[i]
            y_hi = min(cap_i, R // m_i, target_k)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            qmin: Optional[int] = None
            qmax: Optional[int] = None
            y_min = y_max = 0
            if cand is not None:
                y, step = cand
                while y <= y_hi:
                    sub = solve(i + 1, R - y * m_i)
                    if sub is not None and y + sub[0] == target_k:
                        lo_q, _, hi_q, _ = run(i + 1, R - y * m_i)
                        vlo, vhi = y * lm[i] + lo_q, y * lm[i] + hi_q
                        if qmin is None or vlo < qmin:
                            qmin, y_min = vlo, y
                        if qmax is None or vhi > qmax:
                            qmax, y_max = vhi, y
                    y += step
            if qmin is None or qmax is None:  # impossible on a feasible DAG
                raise ValueError("mass not attainable")
            result = (qmin, y_min, qmax, y_max)
            ext[(i, R)] = result
            return result

        run(0, dag["R0"])
        return ext

    def _optimal_edges(self, dag: dict, i: int, R: int,
                       ascending: bool) -> list[tuple[int, int]]:
        """Optimal min-count edges out of ``(i, R)`` as (y, sub_R) pairs.

        The ascending list is computed once per DAG state and cached; the
        descending order is its reverse.
        """
        cache = dag.setdefault("_edges_cache", {})
        cached = cache.get((i, R))
        if cached is None:
            mm, cc = dag["mm"], dag["cc"]
            smax, sg, memo, solve = (dag["smax"], dag["sg"],
                                     dag["memo"], dag["solve"])
            target_k = memo[i][R][0]
            m_i = mm[i]
            y_hi = min(cc[i], R // m_i, target_k)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            edges: list[tuple[int, int]] = []
            if cand is not None:
                y, step = cand
                while y <= y_hi:
                    sub_R = R - y * m_i
                    sub = solve(i + 1, sub_R)
                    if sub is not None and y + sub[0] == target_k:
                        edges.append((y, sub_R))
                    y += step
            cache[(i, R)] = edges
            cached = edges
        return cached if ascending else list(reversed(cached))

    def _find_inside(self, dag: dict, ext: dict, labeled_m: list[int],
                     q_lo: int, q_hi: int,
                     canonical: tuple[int, ...]) -> Optional[tuple[int, ...]]:
        """Return an optimal path with effective labeled sum in [q_lo, q_hi].

        The canonical vector itself does not count.  Suffix extrema prune
        subtrees whose whole labeled-sum range misses the window, so the walk
        never degenerates into enumerating counts.  Returns the vector in
        canonical id order, or ``None``.
        """
        order = dag["order"]
        n = self.n
        lm = [labeled_m[idx] for idx in order]
        prefix: list[int] = []

        def finish() -> tuple[int, ...]:
            vec = [0] * n
            for pos, idx in enumerate(order):
                vec[idx] = self.lo[idx] + (prefix[pos]
                                           if pos < len(prefix) else 0)
            return tuple(vec)

        def walk(i: int, R: int, q0: int) -> Optional[tuple[int, ...]]:
            if R == 0:
                # Remaining coordinates are pinned to effective count 0, so
                # the leaf labeled sum is already fixed at q0.
                if not (q_lo <= q0 <= q_hi):
                    return None
                vec = finish()
                return vec if vec != canonical else None
            self._tick()
            lo_q, _, hi_q, _ = ext[(i, R)]
            if q0 + hi_q < q_lo or q0 + lo_q > q_hi:
                return None  # every suffix labeled sum misses the window
            for y, sub_R in self._optimal_edges(dag, i, R, True):
                prefix.append(y)
                found = walk(i + 1, sub_R, q0 + y * lm[i])
                prefix.pop()
                if found is not None:
                    return found
            return None

        return walk(0, dag["R0"], 0)

    def _find_extreme_outside(self, dag: dict, ext: dict,
                              labeled_m: list[int], bound: int,
                              find_min: bool):
        """Optimal-path leaf with extreme labeled sum on one outside side.

        ``find_min`` -> minimum effective labeled sum ``>= bound``;
        otherwise maximum effective labeled sum ``<= bound``.  Suffix extrema
        prune every subtree that cannot (a) reach the admissible side or
        (b) beat the incumbent.  Returns ``(Q, vector_in_id_order)`` or None.
        """
        order = dag["order"]
        n = self.n
        lm = [labeled_m[idx] for idx in order]
        prefix: list[int] = []
        # Best admissible leaf seen so far as (Q, vector); ties break on the
        # canonical-id-order vector so the answer is deterministic.
        incumbent: Optional[tuple[int, tuple[int, ...]]] = None

        def finish() -> tuple[int, ...]:
            vec = [0] * n
            for pos, idx in enumerate(order):
                vec[idx] = self.lo[idx] + (prefix[pos]
                                           if pos < len(prefix) else 0)
            return tuple(vec)

        def improves(Q: int, vec: tuple[int, ...]) -> bool:
            if incumbent is None:
                return True
            if find_min:
                return (Q, vec) < (incumbent[0], incumbent[1])
            return (-Q, vec) < (-incumbent[0], incumbent[1])

        def walk(i: int, R: int, q0: int):
            nonlocal incumbent
            if R == 0:
                Q = q0
                ok = Q >= bound if find_min else Q <= bound
                if not ok or not improves(Q, finish_vec := finish()):
                    return None
                incumbent = (Q, finish_vec)
                return finish_vec
            self._tick()
            lo_q, _, hi_q, _ = ext[(i, R)]
            slow, shigh = q0 + lo_q, q0 + hi_q
            if find_min:
                if shigh < bound:
                    return None  # subtree cannot reach the admissible side
                # Equality cannot beat the incumbent either, and edge ordering
                # makes the first extremal leaf deterministic -- pruning ties
                # is what keeps this from enumerating co-labeled winners.
                if incumbent is not None and slow >= incumbent[0]:
                    return None
            else:
                if slow > bound:
                    return None
                if incumbent is not None and shigh <= incumbent[0]:
                    return None
            best_vec = None
            for y, sub_R in self._optimal_edges(dag, i, R, find_min):
                prefix.append(y)
                cand_vec = walk(i + 1, sub_R, q0 + y * lm[i])
                prefix.pop()
                if cand_vec is not None:
                    best_vec = cand_vec  # a new incumbent was set
            return best_vec

        return walk(0, dag["R0"], 0)

    def label_audit(self, winning_blocks: list[dict],
                    collected: list[tuple[int, ...]],
                    label_increments: list[int],
                    supplementary_tolerance: int,
                    truncated: bool = False) -> dict:
        """Audit other second-level optima against the labeled canonical winner.

        The two-level optimum (nearest mass, then minimum particle count) is
        taken as already established by :meth:`solve`; ``winning_blocks`` carry
        each optimal mass's minimum-count DAG and ``collected`` the gathered
        winner vectors (canonical id order).

        Given per-component non-negative label mass increments ``d_i`` (at
        least one positive) and a supplementary-peak tolerance ``tau``, the
        canonical (lexicographically smallest) count vector ``x*`` fixes the
        reference labeled mass ``L* = sum x*_i (m_i + d_i)``.  Every *other*
        minimum-count explanation is audited exactly, over the optimal DAG
        edges only (no count interval is expanded):

        * a different vector with ``|L - L*| <= tau`` witnesses that the
          supplementary peak cannot rule it out (``indistinguishable``);
        * otherwise the outside vector whose labeled peak sits closest to the
          tolerance window (minimum ``|L - L*| > tau``) is returned as the
          nearest counterexample candidate (``distinguishable``).

        All arithmetic stays exact (arbitrary-precision integers).
        """
        n = self.n
        assert len(label_increments) == n
        canonical = self._canonical_winner(winning_blocks, collected,
                                           truncated)
        labeled_m = [self.m[i] + label_increments[i] for i in range(n)]
        Lstar = sum(canonical[i] * labeled_m[i] for i in range(n))
        canonical_mass = sum(canonical[i] * self.m[i] for i in range(n))
        lo_offset = sum(self.lo[i] * labeled_m[i] for i in range(n))
        # Effective labeled sums Q satisfy L = Q + lo_offset.
        q_star = Lstar - lo_offset
        q_lo = q_star - supplementary_tolerance
        q_hi = q_star + supplementary_tolerance

        # Closest vector STRICTLY OUTSIDE the window: (|L-L*|, vec).
        outside_best: Optional[tuple[int, tuple[int, ...]]] = None

        for block in winning_blocks:
            dag = block["dag"]
            ext = self._label_extrema(dag, labeled_m)

            witness = self._find_inside(dag, ext, labeled_m, q_lo, q_hi,
                                        canonical)
            if witness is not None:
                Lv = sum(witness[i] * labeled_m[i] for i in range(n))
                return {
                    "conclusion": "indistinguishable",
                    "canonical_vector": canonical,
                    "canonical_total_mass": canonical_mass,
                    "reference_labeled_mass": Lstar,
                    "supplementary_tolerance": supplementary_tolerance,
                    "witness_vector": witness,
                    "witness_labeled_mass": Lv,
                    "witness_labeled_error": Lv - Lstar,
                    "witness_labeled_absolute_error": abs(Lv - Lstar),
                }

            # No inside vector in this block: locate its closest outside leaf.
            up = self._find_extreme_outside(dag, ext, labeled_m,
                                            q_hi + 1, True)
            down = self._find_extreme_outside(dag, ext, labeled_m,
                                              q_lo - 1, False)
            for cand_vec in (up, down):
                if cand_vec is None or cand_vec == canonical:
                    continue
                Lv = sum(cand_vec[i] * labeled_m[i] for i in range(n))
                dist = abs(Lv - Lstar)
                if dist <= supplementary_tolerance:
                    continue  # defensive: extrema searches exclude the window
                if (outside_best is None or dist < outside_best[0]
                        or (dist == outside_best[0] and cand_vec < outside_best[1])):
                    outside_best = (dist, cand_vec)

        assert outside_best is not None, "label audit requires an alternative"
        gap, vec = outside_best
        nearest_mass = sum(vec[i] * labeled_m[i] for i in range(n))
        return {
            "conclusion": "distinguishable",
            "canonical_vector": canonical,
            "canonical_total_mass": canonical_mass,
            "reference_labeled_mass": Lstar,
            "supplementary_tolerance": supplementary_tolerance,
            "nearest_vector": vec,
            "nearest_labeled_mass": nearest_mass,
            "nearest_labeled_error": nearest_mass - Lstar,
            "nearest_labeled_absolute_error": gap,
            "margin_to_tolerance": gap - supplementary_tolerance,
        }

    # ------------------------------------------------------------------ #
    # Top-level driver
    # ------------------------------------------------------------------ #

    def solve(self, max_collect: int) -> dict:
        below = self._extreme(-1)
        above = self._extreme(1)

        extremes: list[tuple[int, tuple[int, ...]]] = []
        if below is not None:
            extremes.append(below)
        if above is not None:
            extremes.append(above)
        if not extremes:
            raise ValueError("empty counting box")  # prevented by validation

        best_dist = min(abs(mass - self.T) for mass, _ in extremes)
        optimal_masses = sorted({mass for mass, _ in extremes
                                 if abs(mass - self.T) == best_dist})
        within = best_dist <= self.tol

        if not within:
            return {
                "status": "unsatisfiable",
                "target": self.T,
                "tolerance": self.tol,
                "within_tolerance": False,
                "best_distance": best_dist,
                "component_order": self.ids,
                "nearest_below": self._witness(below),
                "nearest_above": self._witness(above),
            }

        # A below and an above witness sharing |error| are BOTH optimal
        # masses; the particle-count objective is taken across all of them.
        blocks = []
        for mass in optimal_masses:
            dag = self._build_min_count_dag(mass)
            min_extra, num_vectors = dag["root"]
            vectors = (sorted(self._collect_minimal(dag, max_collect))
                       if dag["R0"] != 0 else [tuple(self.lo)])
            blocks.append({
                "mass": mass,
                "particle_count": sum(self.lo) + min_extra,
                "num_vectors": num_vectors,
                "vectors": vectors,
                "truncated": num_vectors > len(vectors),
                "dag": dag,
            })

        best_particles = min(b["particle_count"] for b in blocks)
        winners: list[tuple[int, ...]] = []
        num_winners = 0
        truncated = False
        winning_blocks: list[dict] = []
        for b in blocks:
            if b["particle_count"] == best_particles:
                winners.extend(b["vectors"])
                num_winners += b["num_vectors"]
                truncated = truncated or b["truncated"]
                winning_blocks.append(b)
        winners.sort()

        return {
            "status": "optimal",
            "target": self.T,
            "tolerance": self.tol,
            "within_tolerance": True,
            "best_distance": best_dist,
            "component_order": self.ids,
            "optimal_masses": optimal_masses,
            "particle_count": best_particles,
            "num_optimal_explanations": num_winners,
            "unique": num_winners == 1,
            "truncated_list": truncated or num_winners > len(winners),
            "vectors": winners,
            "winning_blocks": winning_blocks,
            "nearest_below": self._witness(below),
            "nearest_above": self._witness(above),
        }

    def _witness(self, extreme: Optional[tuple[int, tuple[int, ...]]]) -> Optional[dict]:
        if extreme is None:
            return None
        mass, vec = extreme
        return {
            "total_mass": mass,
            "error": mass - self.T,
            "absolute_error": abs(mass - self.T),
            "particle_count": sum(vec),
            "vector": list(vec),
            "counts": [
                {"id": self.ids[i], "mass": self.m[i], "count": vec[i],
                 "mass_contribution": vec[i] * self.m[i]}
                for i in range(self.n)
            ],
        }


def _congruence_candidates(y_lo: int, y_hi: int, m_i: int, R: int,
                           g2: int) -> Optional[tuple[int, int]]:
    """Smallest y >= y_lo with m_i*y ≡ R (mod g2), together with the step.

    Returns ``(first_y, step)`` or ``None`` when the congruence has no
    solution in the window.  ``g2 <= 1`` imposes no restriction.
    """
    if g2 <= 1:
        return y_lo, 1
    a = m_i % g2
    h = gcd(a, g2)
    if R % h != 0:
        return None
    mod = g2 // h
    if mod == 1:
        return y_lo, 1
    a2 = a // h
    b2 = (R // h) % mod
    inv = pow(a2, -1, mod)
    r0 = (inv * b2) % mod
    first = y_lo + ((r0 - y_lo) % mod)
    if first > y_hi:
        return None
    return first, mod

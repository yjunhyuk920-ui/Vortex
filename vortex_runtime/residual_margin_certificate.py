from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linprog


class MarginCertificateInvariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarginCertificate:
    target_token: int
    coefficients: np.ndarray
    coefficient_sha256: str
    coefficient_l1: float
    coefficient_l2: float
    coefficient_linf: float
    coefficient_support: int
    score_sha256: str
    stable_winner: int
    worst_competitor: int
    minimum_certified_margin: float
    minimum_computed_margin: float
    maximum_score_error_bound: float
    active_constraint_count: int
    active_iterations: int
    full_constraint_fallback_used: bool
    solver_nit: int
    solver_objective: float

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("coefficients")
        return result


@dataclass(frozen=True)
class MarginSolveResult:
    status: str
    target_token: int
    certificate: MarginCertificate | None
    active_constraint_count: int
    active_iterations: int
    full_constraint_fallback_used: bool
    solver_status: int
    solver_message: str
    solver_nit: int
    infeasible_subset_count: int
    infeasible_subset_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_token": self.target_token,
            "certificate": None if self.certificate is None else self.certificate.summary(),
            "active_constraint_count": self.active_constraint_count,
            "active_iterations": self.active_iterations,
            "full_constraint_fallback_used": self.full_constraint_fallback_used,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "solver_nit": self.solver_nit,
            "infeasible_subset_count": self.infeasible_subset_count,
            "infeasible_subset_sha256": self.infeasible_subset_sha256,
        }


@dataclass(frozen=True)
class CertificatePopulationSummary:
    total: int
    certified: int
    infeasible: int
    certificate_fraction: float
    coefficient_l1_p50: float | None
    coefficient_l1_p95: float | None
    coefficient_linf_p50: float | None
    coefficient_linf_p95: float | None
    support_p50: float | None
    support_p95: float | None
    certified_margin_minimum: float | None
    active_constraints_p50: float
    active_constraints_p95: float
    full_fallback_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def stable_descending_indices(scores: np.ndarray) -> np.ndarray:
    vector = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise MarginCertificateInvariantError("score source contains nonfinite values")
    ids = np.arange(vector.size, dtype=np.int64)
    return np.lexsort((ids, -vector))


def stable_argmax(scores: np.ndarray) -> int:
    return int(stable_descending_indices(scores)[0])


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


class ResidualMarginCompiler:
    """Target-seeing FP64 margin compiler over one online residual bank."""

    def __init__(
        self,
        residual_bank: torch.Tensor | np.ndarray,
        *,
        required_unique_margin: float,
        initial_topk_per_score_source: int,
        violator_batch_size: int,
        maximum_active_iterations: int,
        full_constraint_fallback: bool,
        primal_feasibility_tolerance: float,
        dual_feasibility_tolerance: float,
    ) -> None:
        if isinstance(residual_bank, torch.Tensor):
            bank = residual_bank.detach().to(torch.float64).contiguous().cpu().numpy()
        else:
            bank = np.asarray(residual_bank, dtype=np.float64)
        if bank.ndim != 2 or bank.shape[0] <= 0 or bank.shape[1] <= 1:
            raise MarginCertificateInvariantError(
                f"residual bank must be KxV, got {bank.shape}"
            )
        if not np.isfinite(bank).all():
            raise MarginCertificateInvariantError("residual bank contains nonfinite words")
        if float(required_unique_margin) <= 0.0:
            raise MarginCertificateInvariantError("required margin must be positive")
        if min(
            int(initial_topk_per_score_source),
            int(violator_batch_size),
            int(maximum_active_iterations),
        ) <= 0:
            raise MarginCertificateInvariantError("active-set parameters must be positive")
        self.bank = np.ascontiguousarray(bank)
        self.abs_bank = np.abs(self.bank)
        self.k, self.vocab = map(int, self.bank.shape)
        self.margin = float(required_unique_margin)
        self.initial_topk = int(initial_topk_per_score_source)
        self.violator_batch = int(violator_batch_size)
        self.maximum_iterations = int(maximum_active_iterations)
        self.full_fallback = bool(full_constraint_fallback)
        self.primal_tolerance = float(primal_feasibility_tolerance)
        self.dual_tolerance = float(dual_feasibility_tolerance)
        self.objective = np.ones(2 * self.k, dtype=np.float64)
        self.bounds = [(0.0, None)] * (2 * self.k)
        operation_count = 2 * self.k + 4
        unit_roundoff = 2.0**-53
        self.gamma = (operation_count * unit_roundoff) / (
            1.0 - operation_count * unit_roundoff
        )

    def _constraint_matrix(
        self, coarse: np.ndarray, target: int, competitors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        comp = np.asarray(competitors, dtype=np.int64).reshape(-1)
        if comp.size == 0 or bool(np.any(comp == target)):
            raise MarginCertificateInvariantError("invalid competitor set")
        delta = self.bank[:, target : target + 1] - self.bank[:, comp]
        matrix = np.concatenate((-delta.T, delta.T), axis=1)
        rhs = coarse[target] - coarse[comp] - self.margin
        if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
            raise MarginCertificateInvariantError("LP constraints are nonfinite")
        return np.ascontiguousarray(matrix), np.ascontiguousarray(rhs)

    def _solve_subset(
        self, coarse: np.ndarray, target: int, competitors: np.ndarray
    ) -> Any:
        matrix, rhs = self._constraint_matrix(coarse, target, competitors)
        return linprog(
            self.objective,
            A_ub=matrix,
            b_ub=rhs,
            bounds=self.bounds,
            method="highs-ds",
            options={
                "presolve": True,
                "primal_feasibility_tolerance": self.primal_tolerance,
                "dual_feasibility_tolerance": self.dual_tolerance,
            },
        )

    def _finite_word_scan(
        self, coarse: np.ndarray, target: int, coefficients: np.ndarray
    ) -> dict[str, Any]:
        coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if coeff.size != self.k or not np.isfinite(coeff).all():
            raise MarginCertificateInvariantError("invalid coefficient vector")
        scores = coarse + coeff @ self.bank
        absolute_sum = np.abs(coarse) + np.abs(coeff) @ self.abs_bank
        score_error = self.gamma * absolute_sum
        if not np.isfinite(scores).all() or not np.isfinite(score_error).all():
            raise MarginCertificateInvariantError("score enclosure is nonfinite")
        computed_margin = scores[target] - scores
        lower_margin = computed_margin - score_error[target] - score_error
        computed_margin[target] = np.inf
        lower_margin[target] = np.inf
        worst = int(np.argmin(lower_margin))
        winner = stable_argmax(scores)
        return {
            "scores": scores,
            "score_error": score_error,
            "computed_margin": computed_margin,
            "lower_margin": lower_margin,
            "worst_competitor": worst,
            "stable_winner": winner,
            "minimum_computed_margin": float(computed_margin[worst]),
            "minimum_certified_margin": float(lower_margin[worst]),
            "certified": winner == target and float(lower_margin[worst]) > 0.0,
        }

    def _initial_competitors(
        self,
        target: int,
        score_sources: Sequence[torch.Tensor | np.ndarray],
    ) -> np.ndarray:
        if not score_sources:
            raise MarginCertificateInvariantError("no initial score sources")
        selected: set[int] = set()
        for source in score_sources:
            if isinstance(source, torch.Tensor):
                scores = (
                    source.detach()
                    .to(torch.float64)
                    .reshape(-1)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
            else:
                scores = np.asarray(source, dtype=np.float64).reshape(-1)
            if scores.size != self.vocab:
                raise MarginCertificateInvariantError("initial score width mismatch")
            local_count = 0
            for index in stable_descending_indices(scores):
                competitor = int(index)
                if competitor == target:
                    continue
                selected.add(competitor)
                local_count += 1
                if local_count >= self.initial_topk:
                    break
        if not selected:
            selected.add(0 if target != 0 else 1)
        return np.asarray(sorted(selected), dtype=np.int64)

    @staticmethod
    def _subset_evidence(competitors: np.ndarray) -> tuple[int, str]:
        array = np.ascontiguousarray(competitors, dtype=np.int64)
        return int(array.size), array_sha256(array)

    def _infeasible_result(
        self,
        *,
        target: int,
        competitors: np.ndarray,
        iterations: int,
        full_fallback_used: bool,
        solver_result: Any,
    ) -> MarginSolveResult:
        count, digest = self._subset_evidence(competitors)
        return MarginSolveResult(
            status="infeasible",
            target_token=target,
            certificate=None,
            active_constraint_count=count,
            active_iterations=int(iterations),
            full_constraint_fallback_used=bool(full_fallback_used),
            solver_status=int(solver_result.status),
            solver_message=str(solver_result.message),
            solver_nit=int(getattr(solver_result, "nit", 0) or 0),
            infeasible_subset_count=count,
            infeasible_subset_sha256=digest,
        )

    def _certificate_result(
        self,
        *,
        target: int,
        coefficients: np.ndarray,
        scan: dict[str, Any],
        active_count: int,
        iterations: int,
        full_fallback_used: bool,
        solver_result: Any,
    ) -> MarginSolveResult:
        coeff = np.ascontiguousarray(coefficients, dtype=np.float64)
        certificate = MarginCertificate(
            target_token=target,
            coefficients=coeff,
            coefficient_sha256=array_sha256(coeff),
            coefficient_l1=float(np.sum(np.abs(coeff))),
            coefficient_l2=float(np.linalg.vector_norm(coeff, ord=2)),
            coefficient_linf=float(np.max(np.abs(coeff))),
            coefficient_support=int(np.sum(np.abs(coeff) > 1e-12)),
            score_sha256=array_sha256(np.asarray(scan["scores"], dtype=np.float64)),
            stable_winner=int(scan["stable_winner"]),
            worst_competitor=int(scan["worst_competitor"]),
            minimum_certified_margin=float(scan["minimum_certified_margin"]),
            minimum_computed_margin=float(scan["minimum_computed_margin"]),
            maximum_score_error_bound=float(np.max(scan["score_error"])),
            active_constraint_count=int(active_count),
            active_iterations=int(iterations),
            full_constraint_fallback_used=bool(full_fallback_used),
            solver_nit=int(getattr(solver_result, "nit", 0) or 0),
            solver_objective=float(solver_result.fun),
        )
        return MarginSolveResult(
            status="certified",
            target_token=target,
            certificate=certificate,
            active_constraint_count=int(active_count),
            active_iterations=int(iterations),
            full_constraint_fallback_used=bool(full_fallback_used),
            solver_status=int(solver_result.status),
            solver_message=str(solver_result.message),
            solver_nit=int(getattr(solver_result, "nit", 0) or 0),
            infeasible_subset_count=0,
            infeasible_subset_sha256=array_sha256(
                np.empty(0, dtype=np.int64)
            ),
        )

    def solve(
        self,
        coarse_logits: torch.Tensor | np.ndarray,
        target_token: int,
        *,
        initial_score_sources: Sequence[torch.Tensor | np.ndarray],
    ) -> MarginSolveResult:
        if isinstance(coarse_logits, torch.Tensor):
            coarse = (
                coarse_logits.detach()
                .to(torch.float64)
                .reshape(-1)
                .contiguous()
                .cpu()
                .numpy()
            )
        else:
            coarse = np.asarray(coarse_logits, dtype=np.float64).reshape(-1)
        if coarse.size != self.vocab or not np.isfinite(coarse).all():
            raise MarginCertificateInvariantError("invalid coarse logits")
        target = int(target_token)
        if not 0 <= target < self.vocab:
            raise MarginCertificateInvariantError("target outside vocabulary")

        active_set = {
            int(value)
            for value in self._initial_competitors(
                target, initial_score_sources
            ).tolist()
        }
        for iteration in range(1, self.maximum_iterations + 1):
            active = np.asarray(sorted(active_set), dtype=np.int64)
            result = self._solve_subset(coarse, target, active)
            if int(result.status) == 2:
                return self._infeasible_result(
                    target=target,
                    competitors=active,
                    iterations=iteration,
                    full_fallback_used=False,
                    solver_result=result,
                )
            if int(result.status) != 0 or result.x is None:
                raise MarginCertificateInvariantError(
                    f"HiGHS unresolved status {result.status}: {result.message}"
                )
            coefficients = np.asarray(
                result.x[: self.k] - result.x[self.k :], dtype=np.float64
            )
            scan = self._finite_word_scan(coarse, target, coefficients)
            if bool(scan["certified"]):
                return self._certificate_result(
                    target=target,
                    coefficients=coefficients,
                    scan=scan,
                    active_count=int(active.size),
                    iterations=iteration,
                    full_fallback_used=False,
                    solver_result=result,
                )
            lower = np.asarray(scan["lower_margin"], dtype=np.float64)
            added = 0
            for index in np.argsort(lower, kind="stable"):
                competitor = int(index)
                if competitor == target or competitor in active_set:
                    continue
                active_set.add(competitor)
                added += 1
                if added >= self.violator_batch:
                    break
            if added == 0:
                break

        if not self.full_fallback:
            raise MarginCertificateInvariantError(
                "active set unresolved and full fallback disabled"
            )
        competitors = np.delete(
            np.arange(self.vocab, dtype=np.int64), target
        )
        result = self._solve_subset(coarse, target, competitors)
        if int(result.status) == 2:
            return self._infeasible_result(
                target=target,
                competitors=competitors,
                iterations=self.maximum_iterations + 1,
                full_fallback_used=True,
                solver_result=result,
            )
        if int(result.status) != 0 or result.x is None:
            raise MarginCertificateInvariantError(
                f"full HiGHS unresolved status {result.status}: {result.message}"
            )
        coefficients = np.asarray(
            result.x[: self.k] - result.x[self.k :], dtype=np.float64
        )
        scan = self._finite_word_scan(coarse, target, coefficients)
        if not bool(scan["certified"]):
            raise MarginCertificateInvariantError(
                "full LP feasible but complete finite-word scan did not certify"
            )
        return self._certificate_result(
            target=target,
            coefficients=coefficients,
            scan=scan,
            active_count=int(competitors.size),
            iterations=self.maximum_iterations + 1,
            full_fallback_used=True,
            solver_result=result,
        )


def summarize_population(
    results: Sequence[MarginSolveResult],
) -> CertificatePopulationSummary:
    if not results:
        raise MarginCertificateInvariantError("empty certificate population")
    certificates = [
        result.certificate
        for result in results
        if result.certificate is not None
    ]
    l1 = [float(value.coefficient_l1) for value in certificates]
    linf = [float(value.coefficient_linf) for value in certificates]
    support = [float(value.coefficient_support) for value in certificates]
    margins = [
        float(value.minimum_certified_margin) for value in certificates
    ]
    active = [float(result.active_constraint_count) for result in results]
    certified = sum(result.status == "certified" for result in results)
    infeasible = sum(result.status == "infeasible" for result in results)
    return CertificatePopulationSummary(
        total=len(results),
        certified=certified,
        infeasible=infeasible,
        certificate_fraction=certified / len(results),
        coefficient_l1_p50=_percentile(l1, 50),
        coefficient_l1_p95=_percentile(l1, 95),
        coefficient_linf_p50=_percentile(linf, 50),
        coefficient_linf_p95=_percentile(linf, 95),
        support_p50=_percentile(support, 50),
        support_p95=_percentile(support, 95),
        certified_margin_minimum=min(margins) if margins else None,
        active_constraints_p50=float(_percentile(active, 50) or 0.0),
        active_constraints_p95=float(_percentile(active, 95) or 0.0),
        full_fallback_count=sum(
            bool(result.full_constraint_fallback_used) for result in results
        ),
    )


def project_target_resources(
    *,
    inherited_hot_bytes: int,
    block_length: int,
    vocabulary_size: int,
    parameter_count: int = 405_000_000_000,
) -> dict[str, Any]:
    k = int(block_length)
    v = int(vocabulary_size)
    coefficient_bytes = k * k * 8
    token_bytes = k * 4
    margin_bytes = k * 8
    total = int(inherited_hot_bytes) + coefficient_bytes + token_bytes + margin_bytes
    scalar_work = k * v + v
    return {
        "inherited_hot_bytes": int(inherited_hot_bytes),
        "float64_coefficient_block_bytes": coefficient_bytes,
        "target_token_id_bytes": token_bytes,
        "certified_margin_bytes": margin_bytes,
        "total_hot_bytes": total,
        "total_hot_gib": total / float(1 << 30),
        "residual_mac_count_per_token": k * v,
        "complete_argmax_margin_scan_per_token": v,
        "scalar_work_per_token": scalar_work,
        "scalar_work_fraction_of_405b": scalar_work / float(parameter_count),
    }


def model_free_controls() -> dict[str, Any]:
    positive_bank = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    positive_compiler = ResidualMarginCompiler(
        positive_bank,
        required_unique_margin=2.0**-20,
        initial_topk_per_score_source=1,
        violator_batch_size=1,
        maximum_active_iterations=4,
        full_constraint_fallback=True,
        primal_feasibility_tolerance=1e-9,
        dual_feasibility_tolerance=1e-9,
    )
    coarse = torch.zeros(3, dtype=torch.float64)
    positive = positive_compiler.solve(
        coarse,
        1,
        initial_score_sources=(coarse, coarse + positive_bank[0]),
    )
    positive_certified = (
        positive.status == "certified"
        and positive.certificate is not None
        and positive.certificate.stable_winner == 1
        and positive.certificate.minimum_certified_margin > 0.0
    )

    negative_bank = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
    negative_compiler = ResidualMarginCompiler(
        negative_bank,
        required_unique_margin=2.0**-20,
        initial_topk_per_score_source=1,
        violator_batch_size=1,
        maximum_active_iterations=2,
        full_constraint_fallback=True,
        primal_feasibility_tolerance=1e-9,
        dual_feasibility_tolerance=1e-9,
    )
    negative_coarse = torch.tensor([1.0, 0.0], dtype=torch.float64)
    negative = negative_compiler.solve(
        negative_coarse,
        1,
        initial_score_sources=(negative_coarse,),
    )
    negative_infeasible = (
        negative.status == "infeasible" and negative.solver_status == 2
    )

    tie_rule = stable_argmax(
        np.asarray([4.0, 4.0, 3.0], dtype=np.float64)
    ) == 0
    digest_bitflip = False
    if positive.certificate is not None:
        changed = positive.certificate.coefficients.copy()
        changed.view(np.uint64)[0] ^= np.uint64(1)
        digest_bitflip = (
            array_sha256(changed)
            != positive.certificate.coefficient_sha256
        )
    passed = all(
        [positive_certified, negative_infeasible, tie_rule, digest_bitflip]
    )
    return {
        "passed": passed,
        "positive_certified": positive_certified,
        "negative_infeasible": negative_infeasible,
        "stable_tie_rule": tie_rule,
        "coefficient_digest_bitflip_detected": digest_bitflip,
        "positive": positive.summary(),
        "negative": negative.summary(),
    }

"""Local sparse MoE routing and speculative lookahead math."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_EXPERT_NAMES = (
    "logic",
    "security",
    "emotional_field",
    "memory",
    "critic",
    "engineer",
    "researcher",
    "coach",
)


@dataclass(frozen=True)
class MoERouteResult:
    """One sparse router decision."""

    output: np.ndarray
    chosen_experts: tuple[int, ...]
    chosen_names: tuple[str, ...]
    router_probabilities: tuple[float, ...]
    active_expert_count: int
    total_experts: int

    @property
    def active_fraction(self) -> float:
        return self.active_expert_count / self.total_experts if self.total_experts else 0.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output"] = self.output.tolist()
        data["active_fraction"] = self.active_fraction
        return data


class ExpertNode:
    """A small local expert module with deterministic weights."""

    def __init__(self, expert_id: int, dimension: int = 4, name: str | None = None, seed: int = 0):
        self.expert_id = int(expert_id)
        self.name = name or f"expert_{expert_id}"
        rng = np.random.default_rng(seed + expert_id * 9973)
        self.weights = rng.normal(0.0, 0.5, size=(dimension, dimension))
        self.execution_count = 0

    def process(self, tensor: np.ndarray) -> np.ndarray:
        self.execution_count += 1
        return np.dot(tensor, self.weights)


class LocalMoERouter:
    """Gating mechanism that executes only the selected local experts."""

    def __init__(
        self,
        num_experts: int = 4,
        dimension: int = 4,
        *,
        expert_names: Sequence[str] | None = None,
        seed: int = 42,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = int(dimension)
        self.num_experts = int(num_experts)
        names = list(expert_names or DEFAULT_EXPERT_NAMES)
        if len(names) < self.num_experts:
            names.extend(f"expert_{idx}" for idx in range(len(names), self.num_experts))

        self.experts = [
            ExpertNode(idx, dimension=self.dimension, name=names[idx], seed=seed)
            for idx in range(self.num_experts)
        ]
        rng = np.random.default_rng(seed)
        self.gate_weights = rng.normal(0.0, 0.5, size=(self.dimension, self.num_experts))

    def route(self, input_vector: np.ndarray | Sequence[Sequence[float]], k: int = 1) -> MoERouteResult:
        """Route ``input_vector`` through exactly top-k selected experts."""

        vector = np.asarray(input_vector, dtype=np.float64)
        if vector.ndim == 1:
            vector = vector[None, :]
        if vector.ndim != 2 or vector.shape[1] != self.dimension:
            raise ValueError(f"input_vector must be shaped (batch, {self.dimension})")
        if not 1 <= k <= self.num_experts:
            raise ValueError("k must satisfy 1 <= k <= num_experts")

        gate_scores = np.dot(vector, self.gate_weights)
        probabilities = _softmax(gate_scores)
        top_indices = np.argsort(probabilities[0])[-k:][::-1]
        top_probs = probabilities[0, top_indices]
        normalized_top_probs = top_probs / top_probs.sum()

        output = np.zeros_like(vector)
        for expert_idx, weight in zip(top_indices, normalized_top_probs):
            output += float(weight) * self.experts[int(expert_idx)].process(vector)

        return MoERouteResult(
            output=output,
            chosen_experts=tuple(int(idx) for idx in top_indices),
            chosen_names=tuple(self.experts[int(idx)].name for idx in top_indices),
            router_probabilities=tuple(float(probabilities[0, idx]) for idx in top_indices),
            active_expert_count=len(top_indices),
            total_experts=self.num_experts,
        )

    def execution_counts(self) -> dict[str, int]:
        return {expert.name: expert.execution_count for expert in self.experts}


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def speculative_lookahead_tau(acceptance_rate: float, gamma: int) -> float:
    """Expected accepted tokens per verifier pass for speculative lookahead."""

    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    alpha = float(np.clip(acceptance_rate, 0.0, 1.0))
    if alpha == 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def vector_from_dii_summary(summary: dict[str, object]) -> np.ndarray:
    """Build a 4D router vector from DII state for local expert selection."""

    current = float(summary.get("current", 0.5))
    variance = float(summary.get("variance", 0.0))
    samples = float(summary.get("samples", 0))
    awake = 1.0 if summary.get("awake") else 0.0
    return np.array([[current, min(variance * 100.0, 1.0), min(samples / 50.0, 1.0), awake]])


def verify_moe_pipeline() -> bool:
    """Self-check proving local sparse routing and lookahead math."""

    print("[*] Testing Local MoE Equation Validity...")
    try:
        router = LocalMoERouter(num_experts=4, dimension=4, seed=7)
        sample_input = np.array([[0.1, 0.9, -0.4, 0.3]])
        result = router.route(sample_input, k=1)
        counts = router.execution_counts()

        assert result.output.shape == (1, 4), "Output dimension corruption."
        assert result.active_expert_count == 1, "Router executed more than one expert."
        assert sum(counts.values()) == 1, "Execution counters show dense activation."

        tau = speculative_lookahead_tau(acceptance_rate=0.8, gamma=4)
        assert tau > 3.0, "Lookahead did not improve expected verifier throughput."

        print("[+] Local Routing Successful!")
        print(
            f"    Routed to Local Expert #{result.chosen_experts[0]} "
            f"({result.chosen_names[0]}, confidence: {result.router_probabilities[0]:.2%})"
        )
        print(f"    Resulting State Vector: {result.output[0]}")
        print(f"    Active Experts: {result.active_expert_count}/{result.total_experts}")
        print(f"    Speculative tau(gamma=4, alpha=0.8): {tau:.3f}")
        print("[+] Falsifiable Pipeline Verified: True local sparsity achieved.")
        return True
    except AssertionError as exc:
        print(f"[-] Architecture Verification Failed: {exc}")
        return False
    except Exception as exc:
        print(f"[-] Execution Error: {exc}")
        return False


__all__ = [
    "DEFAULT_EXPERT_NAMES",
    "ExpertNode",
    "LocalMoERouter",
    "MoERouteResult",
    "speculative_lookahead_tau",
    "vector_from_dii_summary",
    "verify_moe_pipeline",
]

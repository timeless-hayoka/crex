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
        """
        Return the proportion of available experts that were selected for this routing decision.
        
        Returns:
            float: The ratio active_expert_count / total_experts, or 0.0 if total_experts is zero or falsy.
        """
        return self.active_expert_count / self.total_experts if self.total_experts else 0.0

    def to_dict(self) -> dict[str, object]:
        """
        Return a plain dict representation of the MoERouteResult suitable for JSON serialization.
        
        The returned dict contains all dataclass fields; the `output` NumPy array is converted to a nested Python list and an `active_fraction` key (float) is added.
        
        Returns:
            dict[str, object]: Mapping of field names to their values with `output` as a list and `active_fraction` included.
        """
        data = asdict(self)
        data["output"] = self.output.tolist()
        data["active_fraction"] = self.active_fraction
        return data


class ExpertNode:
    """A small local expert module with deterministic weights."""

    def __init__(self, expert_id: int, dimension: int = 4, name: str | None = None, seed: int = 0):
        """
        Create an expert node with a deterministic random weight matrix and an execution counter.
        
        Parameters:
            expert_id (int): Unique identifier for the expert; used to derive the default name and the RNG seed offset.
            dimension (int): Size of the square weight matrix; weights shape is (dimension, dimension).
            name (str | None): Optional explicit name; if omitted, defaults to "expert_{expert_id}".
            seed (int): Base RNG seed; the expert's RNG is seeded with (seed + expert_id * 9973) to produce deterministic weights.
        
        Notes:
            - Initializes `weights` sampled from a normal distribution with mean 0.0 and stddev 0.5.
            - Sets `execution_count` to 0.
        """
        self.expert_id = int(expert_id)
        self.name = name or f"expert_{expert_id}"
        rng = np.random.default_rng(seed + expert_id * 9973)
        self.weights = rng.normal(0.0, 0.5, size=(dimension, dimension))
        self.execution_count = 0

    def process(self, tensor: np.ndarray) -> np.ndarray:
        """
        Apply this expert's learned linear transform to the provided input tensor.
        
        Parameters:
        	tensor (np.ndarray): Input array whose trailing dimension matches the expert's configured dimension.
        
        Returns:
        	np.ndarray: The result of multiplying `tensor` by the expert's weight matrix.
        
        Notes:
        	This method increments the expert's internal `execution_count` each time it is called.
        """
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
        """
        Initialize the router with a fixed number of deterministic ExpertNode instances and a gate weight matrix.
        
        Parameters:
            num_experts (int): Number of experts to create; must be greater than 0.
            dimension (int): Dimensionality of input and expert weight matrices; must be greater than 0.
            expert_names (Sequence[str] | None): Optional sequence of names for the experts. If provided names are fewer than
                `num_experts`, remaining names are filled as `expert_{index}`. If omitted, DEFAULT_EXPERT_NAMES is used and
                similarly extended if necessary.
            seed (int): Seed for deterministic initialization of expert weights and the gate weight generator.
        
        Raises:
            ValueError: If `num_experts` <= 0 or `dimension` <= 0.
        """
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
        """
        Selects the top-k experts for the (first) input batch row, runs only those experts, and returns the combined routed output and routing metadata.
        
        Parameters:
            input_vector (np.ndarray | Sequence[Sequence[float]]): A 1-D vector (treated as batch size 1) or a 2-D array shaped (batch, dimension); the second dimension must equal the router's configured dimension.
            k (int): Number of experts to select (must satisfy 1 <= k <= num_experts).
        
        Returns:
            MoERouteResult: Contains the routed output, indices and names of the chosen experts, their router probabilities, the active expert count, and the total expert count.
        
        Raises:
            ValueError: If `input_vector` does not have shape (batch, dimension) after normalization, or if `k` is outside the valid range.
        """

        vector = np.asarray(input_vector, dtype=np.float64)
        if vector.ndim == 1:
            vector = vector[None, :]
        if vector.ndim != 2 or vector.shape[1] != self.dimension:
            raise ValueError(f"input_vector must be shaped (batch, {self.dimension})")
        if vector.shape[0] != 1:
            raise ValueError("route only supports batch size 1")
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
        """
        Map expert names to their execution counts.
        
        Returns:
            dict[str, int]: A dictionary where keys are expert names and values are the number of times each expert has been executed.
        """
        return {expert.name: expert.execution_count for expert in self.experts}


def _softmax(scores: np.ndarray) -> np.ndarray:
    """
    Compute row-wise softmax probabilities for a 2D array.
    
    Parameters:
        scores (np.ndarray): 2D array of raw scores with shape (n_rows, n_cols).
    
    Returns:
        np.ndarray: Array of the same shape as `scores` where each row is transformed into
        a probability distribution that sums to 1.
    """
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def speculative_lookahead_tau(acceptance_rate: float, gamma: int) -> float:
    """
    Compute the expected number of accepted tokens per verifier pass for speculative lookahead.
    
    Parameters:
        acceptance_rate (float): Probability of a single token being accepted; values are clipped to the range [0.0, 1.0].
        gamma (int): Number of speculative tokens considered ahead; must be >= 0.
    
    Returns:
        float: Expected number of accepted tokens per verifier pass.
    
    Raises:
        ValueError: If `gamma` is negative.
    """

    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    alpha = float(np.clip(acceptance_rate, 0.0, 1.0))
    if alpha == 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def vector_from_dii_summary(summary: dict[str, object]) -> np.ndarray:
    """
    Construct a 1x4 router feature vector from a DII summary dictionary.
    
    Parameters:
    	summary (dict[str, object]): Source values for the vector. Recognized keys:
    		- "current": numeric, used directly (default 0.5)
    		- "variance": numeric, scaled by 100.0 and clipped to at most 1.0 (default 0.0)
    		- "samples": numeric, scaled by samples/50.0 and clipped to at most 1.0 (default 0)
    		- "awake": truthy/falsy, converted to 1.0 if truthy, 0.0 otherwise
    
    Returns:
    	np.ndarray: A shape (1, 4) array: [[current, variance_component, samples_component, awake_flag]].
    """

    current = float(summary.get("current", 0.5))
    variance = float(summary.get("variance", 0.0))
    samples = float(summary.get("samples", 0))
    awake = 1.0 if summary.get("awake") else 0.0
    return np.array([[current, min(variance * 100.0, 1.0), min(samples / 50.0, 1.0), awake]])


def verify_moe_pipeline() -> bool:
    """
    Run a self-check that validates local sparse routing behavior and the speculative lookahead formula.
    
    Performs deterministic routing of a fixed sample through a LocalMoERouter (k=1) and verifies:
    - the router produced an output with the expected shape,
    - exactly one expert executed (sparse activation),
    - the speculative lookahead value for alpha=0.8 and gamma=4 exceeds 3.0.
    
    The function prints progress and result information. All exceptions are caught; failures are reported and return False.
    
    Returns:
        bool: `True` if all self-check assertions pass, `False` otherwise.
    """

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

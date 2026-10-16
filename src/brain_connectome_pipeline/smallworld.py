"""Classical binary small-worldness with connected degree-preserving nulls."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import pandas as pd

from brain_connectome_pipeline.config import SmallWorldConfig
from brain_connectome_pipeline.graph import ConstructedGraphs


class SmallWorldAnalysisError(ValueError):
    """Raised when small-world inputs or arithmetic are invalid."""


class SmallWorldOutputError(OSError):
    """Raised when small-world outputs cannot be written safely."""


@dataclass(frozen=True)
class NullSample:
    attempt_id: int
    seed: int
    requested_swaps: int
    successful_swaps: int
    status: Literal["accepted", "rejected"]
    reason: str | None
    clustering: float | None
    path_length: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "seed": self.seed,
            "requested_swaps": self.requested_swaps,
            "successful_swaps": self.successful_swaps,
            "status": self.status,
            "reason": self.reason,
            "clustering": self.clustering,
            "path_length": self.path_length,
        }


@dataclass(frozen=True)
class SmallWorldResults:
    status: Literal["available", "unavailable"]
    reason: str | None
    observed_clustering: float | None
    observed_path_length: float | None
    null_clustering_mean: float | None
    null_clustering_std: float | None
    null_path_length_mean: float | None
    null_path_length_std: float | None
    gamma: float | None
    lambda_value: float | None
    sigma: float | None
    samples: tuple[NullSample, ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "observed_clustering": self.observed_clustering,
            "observed_path_length": self.observed_path_length,
            "null_clustering_mean": self.null_clustering_mean,
            "null_clustering_std": self.null_clustering_std,
            "null_path_length_mean": self.null_path_length_mean,
            "null_path_length_std": self.null_path_length_std,
            "gamma": self.gamma,
            "lambda": self.lambda_value,
            "sigma": self.sigma,
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }


def calculate_smallworld_ratios(
    observed_clustering: float,
    observed_path_length: float,
    null_clustering_mean: float,
    null_path_length_mean: float,
) -> tuple[float, float, float]:
    """Return gamma, lambda, and sigma after guarding classical denominators."""
    values = (
        observed_clustering,
        observed_path_length,
        null_clustering_mean,
        null_path_length_mean,
    )
    if not all(math.isfinite(value) for value in values):
        raise SmallWorldAnalysisError("Small-world ratio inputs must be finite.")
    if null_clustering_mean <= 0:
        raise SmallWorldAnalysisError("Mean null clustering must be positive.")
    if observed_path_length <= 0 or null_path_length_mean <= 0:
        raise SmallWorldAnalysisError("Observed and null path lengths must be positive.")
    gamma = observed_clustering / null_clustering_mean
    lambda_value = observed_path_length / null_path_length_mean
    if lambda_value <= 0:
        raise SmallWorldAnalysisError("Small-world lambda must be positive.")
    sigma = gamma / lambda_value
    if not all(math.isfinite(value) for value in (gamma, lambda_value, sigma)):
        raise SmallWorldAnalysisError("Small-world ratios produced a nonfinite value.")
    return gamma, lambda_value, sigma


def _edge_signature(graph: nx.Graph) -> frozenset[frozenset[Any]]:
    return frozenset(frozenset((source, target)) for source, target in graph.edges)


def rewire_connected_null(
    graph: nx.Graph,
    *,
    requested_swaps: int,
    seed: int,
) -> tuple[nx.Graph, int]:
    """Return a genuinely changed connected null with the exact source degree sequence."""
    if graph.is_directed() or graph.is_multigraph():
        raise SmallWorldAnalysisError("Small-world null generation requires a simple graph.")
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        raise SmallWorldAnalysisError("Connected null generation requires a connected graph.")
    if requested_swaps < 1:
        raise SmallWorldAnalysisError("At least one connected double-edge swap is required.")
    original_signature = _edge_signature(graph)
    original_degrees = dict(graph.degree())
    randomized = graph.copy()
    successful_swaps = int(
        nx.connected_double_edge_swap(
            randomized,
            nswap=requested_swaps,
            seed=seed,
        )
    )
    if successful_swaps <= 0:
        raise SmallWorldAnalysisError("No successful degree-preserving swaps were completed.")
    if _edge_signature(randomized) == original_signature:
        raise SmallWorldAnalysisError("Rewiring ended with the unchanged original edge set.")
    if dict(randomized.degree()) != original_degrees:
        raise SmallWorldAnalysisError(
            "Randomized graph did not preserve the exact degree sequence."
        )
    if not nx.is_connected(randomized):
        raise SmallWorldAnalysisError("Randomized graph was disconnected.")
    return randomized, successful_swaps


def analyze_smallworld(
    graphs: ConstructedGraphs,
    config: SmallWorldConfig,
) -> SmallWorldResults:
    """Estimate classical binary sigma using connected degree-preserving edge swaps."""
    if not config.enabled:
        raise SmallWorldAnalysisError("Small-world analysis is disabled in the configuration.")
    graph = graphs.binary_graph
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    base_metadata: dict[str, Any] = {
        "coefficient": "classical binary sigma",
        "null_model": "NetworkX connected_double_edge_swap",
        "degree_sequence_preserved": True,
        "connectivity_required": True,
        "number_of_nodes": node_count,
        "number_of_edges": edge_count,
        "graph_density": graphs.diagnostics.achieved_density,
        "requested_ensemble_size": config.ensemble_size,
        "minimum_valid_samples": config.minimum_valid_samples,
        "swaps_per_edge": config.swaps_per_edge,
        "maximum_ensemble_attempts": config.maximum_ensemble_attempts,
        "seed": config.seed,
    }

    def unavailable(reason: str) -> SmallWorldResults:
        metadata = dict(base_metadata)
        metadata.update({"accepted_samples": 0, "attempted_samples": 0, "rejected_samples": 0})
        return SmallWorldResults(
            "unavailable",
            reason,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            metadata,
            (),
        )

    if node_count < 4:
        return unavailable("Classical connected edge-swap nulls require at least four nodes.")
    if not nx.is_connected(graph):
        return unavailable(
            "Classical whole-graph small-worldness requires a connected observed graph."
        )
    if edge_count < 2:
        return unavailable("Observed graph has too few edges for degree-preserving rewiring.")

    observed_clustering = float(nx.average_clustering(graph))
    observed_path = float(nx.average_shortest_path_length(graph))
    requested_swaps = max(1, config.swaps_per_edge * edge_count)
    samples: list[NullSample] = []
    accepted_clustering: list[float] = []
    accepted_paths: list[float] = []
    for attempt in range(1, config.maximum_ensemble_attempts + 1):
        if len(accepted_clustering) >= config.ensemble_size:
            break
        sample_seed = config.seed + attempt - 1
        successful_swaps = 0
        try:
            randomized, successful_swaps = rewire_connected_null(
                graph,
                requested_swaps=requested_swaps,
                seed=sample_seed,
            )
            clustering = float(nx.average_clustering(randomized))
            path_length = float(nx.average_shortest_path_length(randomized))
            if not math.isfinite(clustering) or not math.isfinite(path_length):
                samples.append(
                    NullSample(
                        attempt,
                        sample_seed,
                        requested_swaps,
                        successful_swaps,
                        "rejected",
                        "Randomized graph metrics were nonfinite.",
                        None,
                        None,
                    )
                )
                continue
            accepted_clustering.append(clustering)
            accepted_paths.append(path_length)
            samples.append(
                NullSample(
                    attempt,
                    sample_seed,
                    requested_swaps,
                    successful_swaps,
                    "accepted",
                    None,
                    clustering,
                    path_length,
                )
            )
        except (nx.NetworkXException, SmallWorldAnalysisError) as exc:
            samples.append(
                NullSample(
                    attempt,
                    sample_seed,
                    requested_swaps,
                    successful_swaps,
                    "rejected",
                    f"Connected edge swapping failed: {exc}",
                    None,
                    None,
                )
            )

    accepted_count = len(accepted_clustering)
    clustering_mean = statistics.fmean(accepted_clustering) if accepted_count else None
    path_mean = statistics.fmean(accepted_paths) if accepted_count else None
    clustering_std = statistics.stdev(accepted_clustering) if accepted_count >= 2 else None
    path_std = statistics.stdev(accepted_paths) if accepted_count >= 2 else None
    status: Literal["available", "unavailable"] = "available"
    reason = None
    gamma = lambda_value = sigma = None
    if accepted_count < config.minimum_valid_samples:
        status = "unavailable"
        reason = (
            f"Only {accepted_count} valid null graphs were accepted; "
            f"{config.minimum_valid_samples} required."
        )
    else:
        assert clustering_mean is not None and path_mean is not None
        try:
            gamma, lambda_value, sigma = calculate_smallworld_ratios(
                observed_clustering, observed_path, clustering_mean, path_mean
            )
        except SmallWorldAnalysisError as exc:
            status = "unavailable"
            reason = str(exc)
    warnings: list[str] = [
        "Classical sigma depends on graph size, density, degree sequence, and null-model settings."
    ]
    if accepted_count < config.ensemble_size:
        warnings.append(
            f"Accepted {accepted_count} of {config.ensemble_size} requested null graphs."
        )
    metadata = dict(base_metadata)
    metadata.update(
        {
            "accepted_samples": accepted_count,
            "attempted_samples": len(samples),
            "rejected_samples": len(samples) - accepted_count,
            "requested_swaps_per_sample": requested_swaps,
            "null_standard_deviation": "sample standard deviation (n-1 denominator)",
        }
    )
    return SmallWorldResults(
        status,
        reason,
        observed_clustering,
        observed_path,
        clustering_mean,
        clustering_std,
        path_mean,
        path_std,
        gamma,
        lambda_value,
        sigma,
        tuple(samples),
        metadata,
        tuple(warnings),
    )


def write_smallworld_outputs(
    results: SmallWorldResults,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write the summary and any attempted null samples with collision protection."""
    directory = Path(output_dir).resolve()
    paths = {"smallworld_summary": directory / "smallworld_summary.json"}
    if results.samples:
        paths["smallworld_nulls"] = directory / "smallworld_nulls.tsv"
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise SmallWorldOutputError(
            f"Refusing to overwrite existing small-world output(s): {names}"
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        paths["smallworld_summary"].write_text(
            json.dumps(results.summary_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if results.samples:
            pd.DataFrame([sample.to_dict() for sample in results.samples]).to_csv(
                paths["smallworld_nulls"], sep="\t", index=False, float_format="%.17g"
            )
    except (OSError, ValueError) as exc:
        raise SmallWorldOutputError(
            f"Could not write small-world outputs to '{directory}': {exc}"
        ) from exc
    return paths

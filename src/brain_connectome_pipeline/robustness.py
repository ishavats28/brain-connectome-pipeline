"""Targeted-versus-random cumulative node-removal experiments."""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import pandas as pd

from brain_connectome_pipeline.config import RobustnessConfig
from brain_connectome_pipeline.graph import ConstructedGraphs
from brain_connectome_pipeline.models import ConnectomeData


class RobustnessAnalysisError(ValueError):
    """Raised when robustness inputs or calculations are invalid."""


class RobustnessOutputError(OSError):
    """Raised when robustness outputs cannot be written safely."""


@dataclass(frozen=True)
class RemovalCheckpoint:
    checkpoint_index: int
    removed_count: int
    actual_removed_fraction: float
    requested_fractions: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_index": self.checkpoint_index,
            "removed_count": self.removed_count,
            "actual_removed_fraction": self.actual_removed_fraction,
            "requested_fractions": list(self.requested_fractions),
        }


@dataclass(frozen=True)
class RankingRow:
    rank: int
    roi_id: str
    label: str | None
    criterion: Literal["strength", "degree"]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "roi_id": self.roi_id,
            "label": self.label,
            "criterion": self.criterion,
            "score": self.score,
        }


@dataclass(frozen=True)
class RobustnessTrialRow:
    strategy: Literal["targeted", "random"]
    trial_id: str
    trial_seed: int | None
    checkpoint_index: int
    requested_fractions: tuple[float, ...]
    removed_count: int
    actual_removed_fraction: float
    removed_nodes: tuple[str, ...]
    retained_node_fraction: float
    largest_component_fraction: float
    connected_components: int
    survivor_efficiency: float | None
    survivor_efficiency_status: Literal["available", "unavailable"]
    survivor_efficiency_reason: str | None
    original_normalized_efficiency: float | None
    original_normalized_efficiency_status: Literal["available", "unavailable"]
    original_normalized_efficiency_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "trial_id": self.trial_id,
            "trial_seed": self.trial_seed,
            "checkpoint_index": self.checkpoint_index,
            "requested_fractions": json.dumps(list(self.requested_fractions)),
            "removed_count": self.removed_count,
            "actual_removed_fraction": self.actual_removed_fraction,
            "removed_nodes": json.dumps(list(self.removed_nodes)),
            "retained_node_fraction": self.retained_node_fraction,
            "largest_component_fraction": self.largest_component_fraction,
            "connected_components": self.connected_components,
            "global_efficiency_survivor_normalized": self.survivor_efficiency,
            "global_efficiency_survivor_normalized_status": self.survivor_efficiency_status,
            "global_efficiency_survivor_normalized_reason": self.survivor_efficiency_reason,
            "global_efficiency_original_node_normalized": self.original_normalized_efficiency,
            "global_efficiency_original_node_normalized_status": (
                self.original_normalized_efficiency_status
            ),
            "global_efficiency_original_node_normalized_reason": (
                self.original_normalized_efficiency_reason
            ),
        }


@dataclass(frozen=True)
class RobustnessAggregateRow:
    strategy: Literal["targeted", "random_mean"]
    checkpoint_index: int
    requested_fractions: tuple[float, ...]
    removed_count: int
    actual_removed_fraction: float
    metric: str
    value: float | None
    sample_standard_deviation: float | None
    status: Literal["available", "unavailable"]
    reason: str | None
    contributing_trials: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "checkpoint_index": self.checkpoint_index,
            "requested_fractions": list(self.requested_fractions),
            "removed_count": self.removed_count,
            "actual_removed_fraction": self.actual_removed_fraction,
            "metric": self.metric,
            "value": self.value,
            "sample_standard_deviation": self.sample_standard_deviation,
            "status": self.status,
            "reason": self.reason,
            "contributing_trials": self.contributing_trials,
        }

    def to_tsv_dict(self) -> dict[str, Any]:
        row = self.to_dict()
        row["requested_fractions"] = json.dumps(row["requested_fractions"])
        return row


@dataclass(frozen=True)
class RobustnessResults:
    status: Literal["available", "unavailable"]
    reason: str | None
    checkpoints: tuple[RemovalCheckpoint, ...]
    ranking: tuple[RankingRow, ...]
    trial_rows: tuple[RobustnessTrialRow, ...]
    aggregate_rows: tuple[RobustnessAggregateRow, ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "targeted_ranking": [row.to_dict() for row in self.ranking],
            "aggregate_results": [row.to_dict() for row in self.aggregate_rows],
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }


def removal_checkpoints(
    node_count: int, fractions: tuple[float, ...]
) -> tuple[RemovalCheckpoint, ...]:
    """Convert fractions with half-up rounding and explicitly combine duplicate counts."""
    if node_count < 0:
        raise RobustnessAnalysisError("Original node count cannot be negative.")
    grouped: dict[int, list[float]] = {}
    for fraction in fractions:
        count = min(node_count, math.floor(fraction * node_count + 0.5))
        grouped.setdefault(count, []).append(fraction)
    return tuple(
        RemovalCheckpoint(
            index,
            count,
            count / node_count if node_count else 0.0,
            tuple(requested),
        )
        for index, (count, requested) in enumerate(grouped.items())
    )


def _fixed_initial_ranking(
    graphs: ConstructedGraphs,
    connectome: ConnectomeData,
    criterion: Literal["strength", "degree"],
) -> tuple[RankingRow, ...]:
    scores: dict[str, float] = {}
    for node in connectome.roi_ids:
        if criterion == "strength":
            try:
                score = math.fsum(
                    float(attributes["weight"])
                    for attributes in graphs.weighted_graph[node].values()
                )
            except (KeyError, OverflowError) as exc:
                raise RobustnessAnalysisError(
                    f"Could not calculate finite initial strength for node {node!r}."
                ) from exc
        else:
            score = float(graphs.binary_graph.degree(node))
        if not math.isfinite(score):
            raise RobustnessAnalysisError(
                f"Initial {criterion} ranking score is nonfinite for node {node!r}."
            )
        scores[node] = score
    order = {node: index for index, node in enumerate(connectome.roi_ids)}
    ranked = sorted(connectome.roi_ids, key=lambda node: (-scores[node], order[node]))
    return tuple(
        RankingRow(
            rank,
            node,
            connectome.roi_labels[order[node]],
            criterion,
            scores[node],
        )
        for rank, node in enumerate(ranked, start=1)
    )


def _evaluate_sequence(
    graph: nx.Graph,
    sequence: tuple[str, ...],
    checkpoints: tuple[RemovalCheckpoint, ...],
    *,
    strategy: Literal["targeted", "random"],
    trial_id: str,
    trial_seed: int | None,
    original_node_count: int,
) -> list[RobustnessTrialRow]:
    rows: list[RobustnessTrialRow] = []
    original_denominator = original_node_count * (original_node_count - 1)
    for checkpoint in checkpoints:
        removed = sequence[: checkpoint.removed_count]
        survivors = sequence[checkpoint.removed_count :]
        survivor_graph = graph.subgraph(survivors)
        survivor_count = len(survivors)
        components = nx.number_connected_components(survivor_graph) if survivor_count else 0
        largest = max(
            (len(component) for component in nx.connected_components(survivor_graph)),
            default=0,
        )
        if survivor_count >= 2:
            survivor_efficiency = float(nx.global_efficiency(survivor_graph))
            survivor_status: Literal["available", "unavailable"] = "available"
            survivor_reason = None
            efficiency_sum = survivor_efficiency * survivor_count * (survivor_count - 1)
        else:
            survivor_efficiency = None
            survivor_status = "unavailable"
            survivor_reason = "Survivor-normalized efficiency requires at least two nodes."
            efficiency_sum = 0.0
        if original_denominator:
            fixed_efficiency = efficiency_sum / original_denominator
            fixed_status: Literal["available", "unavailable"] = "available"
            fixed_reason = None
        else:
            fixed_efficiency = None
            fixed_status = "unavailable"
            fixed_reason = "Original-node-normalized efficiency requires at least two nodes."
        for value in (survivor_efficiency, fixed_efficiency):
            if value is not None and not math.isfinite(value):
                raise RobustnessAnalysisError("Robustness efficiency calculation was nonfinite.")
        rows.append(
            RobustnessTrialRow(
                strategy,
                trial_id,
                trial_seed,
                checkpoint.checkpoint_index,
                checkpoint.requested_fractions,
                checkpoint.removed_count,
                checkpoint.actual_removed_fraction,
                removed,
                survivor_count / original_node_count,
                largest / original_node_count,
                components,
                survivor_efficiency,
                survivor_status,
                survivor_reason,
                fixed_efficiency,
                fixed_status,
                fixed_reason,
            )
        )
    return rows


def _metric_value(row: RobustnessTrialRow, metric: str) -> tuple[float | None, str | None]:
    if metric == "retained_node_fraction":
        return row.retained_node_fraction, None
    if metric == "largest_component_fraction":
        return row.largest_component_fraction, None
    if metric == "connected_components":
        return float(row.connected_components), None
    if metric == "global_efficiency_survivor_normalized":
        return row.survivor_efficiency, row.survivor_efficiency_reason
    if metric == "global_efficiency_original_node_normalized":
        return row.original_normalized_efficiency, row.original_normalized_efficiency_reason
    raise RobustnessAnalysisError(f"Unknown robustness metric: {metric}")


def _aggregate(
    trial_rows: tuple[RobustnessTrialRow, ...],
    checkpoints: tuple[RemovalCheckpoint, ...],
) -> tuple[RobustnessAggregateRow, ...]:
    metrics = (
        "retained_node_fraction",
        "largest_component_fraction",
        "global_efficiency_survivor_normalized",
        "global_efficiency_original_node_normalized",
        "connected_components",
    )
    aggregated: list[RobustnessAggregateRow] = []
    for checkpoint in checkpoints:
        for strategy, output_strategy in (("targeted", "targeted"), ("random", "random_mean")):
            matching = [
                row
                for row in trial_rows
                if row.strategy == strategy and row.checkpoint_index == checkpoint.checkpoint_index
            ]
            for metric in metrics:
                pairs = [_metric_value(row, metric) for row in matching]
                values = [value for value, _ in pairs if value is not None]
                if values:
                    value = statistics.fmean(values)
                    standard_deviation = (
                        statistics.stdev(values)
                        if strategy == "random" and len(values) >= 2
                        else None
                    )
                    status: Literal["available", "unavailable"] = "available"
                    reason = None
                else:
                    value = standard_deviation = None
                    status = "unavailable"
                    reasons = [reason for _, reason in pairs if reason]
                    reason = reasons[0] if reasons else "Metric was unavailable for all trials."
                aggregated.append(
                    RobustnessAggregateRow(
                        output_strategy,  # type: ignore[arg-type]
                        checkpoint.checkpoint_index,
                        checkpoint.requested_fractions,
                        checkpoint.removed_count,
                        checkpoint.actual_removed_fraction,
                        metric,
                        value,
                        standard_deviation,
                        status,
                        reason,
                        len(values),
                    )
                )
    return tuple(aggregated)


def analyze_robustness(
    graphs: ConstructedGraphs,
    connectome: ConnectomeData,
    config: RobustnessConfig,
) -> RobustnessResults:
    """Compare fixed initial targeted removal with nested random permutations."""
    if not config.enabled:
        raise RobustnessAnalysisError("Robustness analysis is disabled in the configuration.")
    if tuple(graphs.binary_graph.nodes) != connectome.roi_ids:
        raise RobustnessAnalysisError("Graph node order must match the original ROI order.")
    node_count = connectome.number_of_rois
    metadata = {
        "analysis": "mathematical graph perturbation",
        "original_node_count": node_count,
        "initial_requested_density": graphs.diagnostics.requested_density,
        "initial_achieved_density": graphs.diagnostics.achieved_density,
        "initial_retained_edges": graphs.diagnostics.retained_edges,
        "ranking_criterion": config.ranking_criterion,
        "ranking_policy": "fixed once on the intact graph; descending score",
        "ranking_tie_handling": "original ROI order",
        "node_count_rounding": "floor(fraction * original_nodes + 0.5)",
        "random_trials": config.random_trials,
        "base_seed": config.seed,
        "trial_seed_derivation": "base_seed + zero-based trial index",
        "random_sequences": "one permutation per trial; cumulative prefixes are removed",
        "largest_component_normalization": "original node count",
        "survivor_efficiency_denominator": "surviving ordered node pairs",
        "fixed_efficiency_denominator": "original ordered node pairs",
    }
    if node_count < 2:
        return RobustnessResults(
            "unavailable",
            "Robustness comparison requires an original graph with at least two nodes.",
            (),
            (),
            (),
            (),
            metadata,
            (),
        )
    checkpoints = removal_checkpoints(node_count, config.removal_fractions)
    ranking = _fixed_initial_ranking(graphs, connectome, config.ranking_criterion)
    targeted_sequence = tuple(row.roi_id for row in ranking)
    rows = _evaluate_sequence(
        graphs.binary_graph,
        targeted_sequence,
        checkpoints,
        strategy="targeted",
        trial_id="targeted",
        trial_seed=None,
        original_node_count=node_count,
    )
    trial_seeds: list[int] = []
    nodes = list(connectome.roi_ids)
    for trial_index in range(config.random_trials):
        trial_seed = config.seed + trial_index
        trial_seeds.append(trial_seed)
        sequence = nodes.copy()
        random.Random(trial_seed).shuffle(sequence)
        rows.extend(
            _evaluate_sequence(
                graphs.binary_graph,
                tuple(sequence),
                checkpoints,
                strategy="random",
                trial_id=f"random_{trial_index + 1:03d}",
                trial_seed=trial_seed,
                original_node_count=node_count,
            )
        )
    trial_rows = tuple(rows)
    aggregate_rows = _aggregate(trial_rows, checkpoints)
    warnings: list[str] = [
        "Node removal is a mathematical graph perturbation, not a biological lesion model."
    ]
    duplicates = [
        checkpoint for checkpoint in checkpoints if len(checkpoint.requested_fractions) > 1
    ]
    if duplicates:
        warnings.append(
            "Small-network rounding produced duplicate removal counts; "
            "matching fractions were combined."
        )
    score_counts: dict[float, int] = {}
    for row in ranking:
        score_counts[row.score] = score_counts.get(row.score, 0) + 1
    if any(count > 1 for count in score_counts.values()):
        warnings.append("Targeted ranking ties were resolved by original ROI order.")
    if not graphs.diagnostics.connected:
        warnings.append("The intact analysis graph is disconnected.")
    metadata["trial_seeds"] = trial_seeds
    metadata["deduplicated_checkpoint_count"] = len(checkpoints)
    return RobustnessResults(
        "available",
        None,
        checkpoints,
        ranking,
        trial_rows,
        aggregate_rows,
        metadata,
        tuple(warnings),
    )


def write_robustness_outputs(
    results: RobustnessResults,
    output_dir: str | Path,
    *,
    plots_enabled: bool,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write strict robustness metadata and available tables/diagnostic plots."""
    directory = Path(output_dir).resolve()
    paths = {"robustness_summary_json": directory / "robustness_summary.json"}
    if results.status == "available":
        paths.update(
            {
                "robustness_trials": directory / "robustness_trials.tsv",
                "robustness_summary": directory / "robustness_summary.tsv",
                "robustness_ranking": directory / "robustness_ranking.tsv",
            }
        )
        if plots_enabled:
            paths["robustness_lcc_plot"] = directory / "robustness_largest_component.png"
            paths["robustness_efficiency_plot"] = directory / "robustness_fixed_efficiency.png"
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise RobustnessOutputError(f"Refusing to overwrite existing robustness output(s): {names}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        paths["robustness_summary_json"].write_text(
            json.dumps(results.summary_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if results.status == "available":
            pd.DataFrame([row.to_dict() for row in results.trial_rows]).to_csv(
                paths["robustness_trials"], sep="\t", index=False, float_format="%.17g"
            )
            pd.DataFrame([row.to_tsv_dict() for row in results.aggregate_rows]).to_csv(
                paths["robustness_summary"], sep="\t", index=False, float_format="%.17g"
            )
            pd.DataFrame([row.to_dict() for row in results.ranking]).to_csv(
                paths["robustness_ranking"], sep="\t", index=False, float_format="%.17g"
            )
            if plots_enabled:
                from brain_connectome_pipeline.plots import plot_robustness

                plot_robustness(
                    results,
                    paths["robustness_lcc_plot"],
                    paths["robustness_efficiency_plot"],
                )
    except (OSError, ValueError) as exc:
        raise RobustnessOutputError(
            f"Could not write robustness outputs to '{directory}': {exc}"
        ) from exc
    return paths

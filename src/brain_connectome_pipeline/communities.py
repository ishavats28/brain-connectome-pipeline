"""Weighted Louvain communities and module-aware nodal measurements."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd

from brain_connectome_pipeline.config import CommunityConfig
from brain_connectome_pipeline.graph import ConstructedGraphs
from brain_connectome_pipeline.models import ConnectomeData


class CommunityAnalysisError(ValueError):
    """Raised when community analysis inputs or partitions are invalid."""


class CommunityOutputError(OSError):
    """Raised when community outputs cannot be written safely."""


@dataclass(frozen=True)
class CommunityAssignment:
    roi_index: int
    roi_id: str
    label: str | None
    community_id: int
    community_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_index": self.roi_index,
            "roi_id": self.roi_id,
            "label": self.label,
            "community_id": self.community_id,
            "community_size": self.community_size,
        }


@dataclass(frozen=True)
class CommunityResults:
    assignments: tuple[CommunityAssignment, ...]
    partition: dict[str, int]
    modularity: float | None
    modularity_status: Literal["available", "unavailable"]
    modularity_reason: str | None
    metadata: dict[str, Any]
    warnings: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": "available",
            "algorithm": self.metadata["algorithm"],
            "number_of_communities": self.metadata["number_of_communities"],
            "community_sizes": self.metadata["community_sizes"],
            "modularity": self.modularity,
            "modularity_status": self.modularity_status,
            "modularity_reason": self.modularity_reason,
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CommunityNodeMetrics:
    roi_index: int
    roi_id: str
    label: str | None
    community_id: int
    binary_participation: float | None
    binary_participation_status: Literal["available", "unavailable"]
    binary_participation_reason: str | None
    weighted_participation: float | None
    weighted_participation_status: Literal["available", "unavailable"]
    weighted_participation_reason: str | None
    within_module_degree: int
    within_module_strength: float
    within_module_degree_zscore: float | None
    within_module_degree_zscore_status: Literal["available", "unavailable"]
    within_module_degree_zscore_reason: str | None
    within_module_strength_zscore: float | None
    within_module_strength_zscore_status: Literal["available", "unavailable"]
    within_module_strength_zscore_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_index": self.roi_index,
            "roi_id": self.roi_id,
            "label": self.label,
            "community_id": self.community_id,
            "binary_participation": self.binary_participation,
            "binary_participation_status": self.binary_participation_status,
            "binary_participation_reason": self.binary_participation_reason,
            "weighted_participation": self.weighted_participation,
            "weighted_participation_status": self.weighted_participation_status,
            "weighted_participation_reason": self.weighted_participation_reason,
            "within_module_degree": self.within_module_degree,
            "within_module_strength": self.within_module_strength,
            "within_module_degree_zscore": self.within_module_degree_zscore,
            "within_module_degree_zscore_status": self.within_module_degree_zscore_status,
            "within_module_degree_zscore_reason": self.within_module_degree_zscore_reason,
            "within_module_strength_zscore": self.within_module_strength_zscore,
            "within_module_strength_zscore_status": self.within_module_strength_zscore_status,
            "within_module_strength_zscore_reason": self.within_module_strength_zscore_reason,
        }


def validate_partition(graph: nx.Graph, partition: dict[str, int]) -> None:
    graph_nodes = set(graph.nodes)
    partition_nodes = set(partition)
    if graph_nodes != partition_nodes:
        missing = sorted(graph_nodes - partition_nodes)
        extra = sorted(partition_nodes - graph_nodes)
        raise CommunityAnalysisError(
            f"Partition must cover every graph node exactly once; missing={missing}, extra={extra}."
        )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in partition.values()
    ):
        raise CommunityAnalysisError("Community identifiers must be nonnegative integers.")


def detect_communities(
    graphs: ConstructedGraphs,
    connectome: ConnectomeData,
    config: CommunityConfig,
) -> CommunityResults:
    """Detect positive-strength Louvain communities with stable ROI-order identifiers."""
    if not config.enabled:
        raise CommunityAnalysisError("Community analysis is disabled in the configuration.")
    graph = graphs.weighted_graph
    if tuple(graph.nodes) != connectome.roi_ids:
        raise CommunityAnalysisError("Weighted graph node order must match the original ROI order.")
    order = {node: index for index, node in enumerate(connectome.roi_ids)}
    if graph.number_of_edges() == 0:
        raw_communities = [{node} for node in connectome.roi_ids]
        modularity = None
        modularity_status: Literal["available", "unavailable"] = "unavailable"
        modularity_reason = "Weighted modularity is undefined when total retained weight is zero."
    else:
        raw_communities = list(
            nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=config.resolution,
                seed=config.seed,
            )
        )
        modularity = float(
            nx.community.modularity(
                graph,
                raw_communities,
                weight="weight",
                resolution=config.resolution,
            )
        )
        if not math.isfinite(modularity):
            raise CommunityAnalysisError(
                "Weighted modularity calculation returned a nonfinite value."
            )
        modularity_status = "available"
        modularity_reason = None
    communities = sorted(raw_communities, key=lambda nodes: min(order[node] for node in nodes))
    partition = {
        node: community_id
        for community_id, nodes in enumerate(communities)
        for node in sorted(nodes, key=order.__getitem__)
    }
    validate_partition(graph, partition)
    sizes = {community_id: len(nodes) for community_id, nodes in enumerate(communities)}
    assignments = tuple(
        CommunityAssignment(
            index,
            roi_id,
            connectome.roi_labels[index],
            partition[roi_id],
            sizes[partition[roi_id]],
        )
        for index, roi_id in enumerate(connectome.roi_ids)
    )
    warnings: list[str] = [
        "Louvain optimization is heuristic and can vary with seed or resolution."
    ]
    connected = nx.is_connected(graph) if graph.number_of_nodes() else False
    if not connected:
        warnings.append("The weighted graph is disconnected; communities span separate components.")
    if graph.number_of_edges() == 0:
        warnings.append("The zero-edge graph was assigned deterministic singleton communities.")
    metadata = {
        "algorithm": "NetworkX Louvain weighted modularity",
        "weight_attribute": "weight",
        "distance_attribute_used": False,
        "resolution": config.resolution,
        "seed": config.seed,
        "graph_density": graphs.diagnostics.achieved_density,
        "connected": connected,
        "number_of_communities": len(communities),
        "community_sizes": [sizes[index] for index in range(len(communities))],
        "community_id_rule": "ascending minimum original ROI index",
    }
    return CommunityResults(
        assignments,
        partition,
        modularity,
        modularity_status,
        modularity_reason,
        metadata,
        tuple(warnings),
    )


def _participation(
    graph: nx.Graph,
    node: str,
    partition: dict[str, int],
    *,
    weight: str | None,
) -> tuple[float | None, str | None]:
    grouped_values: dict[int, list[float]] = {}
    all_values: list[float] = []
    for neighbor, attributes in graph[node].items():
        value = 1.0 if weight is None else float(attributes[weight])
        if not math.isfinite(value) or value <= 0:
            raise CommunityAnalysisError("Participation requires finite positive edge strengths.")
        community_id = partition[neighbor]
        grouped_values.setdefault(community_id, []).append(value)
        all_values.append(value)
    try:
        total = math.fsum(all_values)
        totals = [math.fsum(values) for values in grouped_values.values()]
    except OverflowError as exc:
        raise CommunityAnalysisError(
            "Participation strength summation exceeded finite floating-point range."
        ) from exc
    if total <= 0:
        kind = "degree" if weight is None else "positive strength"
        return None, f"Participation is undefined for a node with zero {kind}."
    if not math.isfinite(total) or any(not math.isfinite(value) for value in totals):
        raise CommunityAnalysisError("Participation totals must remain finite.")
    value = 1.0 - math.fsum((community_total / total) ** 2 for community_total in totals)
    tolerance = 1e-12
    if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
        raise CommunityAnalysisError(
            f"Participation coefficient outside mathematical bounds: {value}"
        )
    return min(1.0, max(0.0, value)), None


def compute_community_node_metrics(
    graphs: ConstructedGraphs,
    connectome: ConnectomeData,
    communities: CommunityResults,
) -> tuple[CommunityNodeMetrics, ...]:
    """Compute participation and population within-module z-scores in ROI order."""
    validate_partition(graphs.binary_graph, communities.partition)
    within_degree: dict[str, int] = {}
    within_strength: dict[str, float] = {}
    for node in connectome.roi_ids:
        own = communities.partition[node]
        same_module = [
            neighbor
            for neighbor in graphs.binary_graph.neighbors(node)
            if communities.partition[neighbor] == own
        ]
        within_degree[node] = len(same_module)
        try:
            strength = math.fsum(
                float(graphs.weighted_graph[node][neighbor]["weight"]) for neighbor in same_module
            )
        except OverflowError as exc:
            raise CommunityAnalysisError(
                f"Within-module strength overflowed for node {node!r}."
            ) from exc
        if not math.isfinite(strength):
            raise CommunityAnalysisError(f"Within-module strength is nonfinite for node {node!r}.")
        within_strength[node] = strength
    module_members: dict[int, list[str]] = {}
    for node in connectome.roi_ids:
        module_members.setdefault(communities.partition[node], []).append(node)

    def zscore(node: str, values: dict[str, float | int]) -> tuple[float | None, str | None]:
        members = module_members[communities.partition[node]]
        if len(members) < 2:
            return None, "Within-module z-score requires at least two community members."
        sample = np.asarray([values[member] for member in members], dtype=np.float64)
        if not np.isfinite(sample).all():
            raise CommunityAnalysisError("Within-module values must remain finite.")
        scale = float(np.max(np.abs(sample)))
        if scale == 0:
            return None, "Within-module population standard deviation is zero."
        scaled = sample / scale
        standard_deviation = float(np.std(scaled, ddof=0))
        if standard_deviation <= np.finfo(np.float64).eps:
            return None, "Within-module population standard deviation is numerically zero."
        value = (float(values[node]) / scale - float(np.mean(scaled))) / standard_deviation
        if not math.isfinite(value):
            raise CommunityAnalysisError("Within-module z-score calculation was nonfinite.")
        return value, None

    rows: list[CommunityNodeMetrics] = []
    for index, node in enumerate(connectome.roi_ids):
        binary_participation, binary_reason = _participation(
            graphs.binary_graph, node, communities.partition, weight=None
        )
        weighted_participation, weighted_reason = _participation(
            graphs.weighted_graph, node, communities.partition, weight="weight"
        )
        degree_z, degree_z_reason = zscore(node, within_degree)
        strength_z, strength_z_reason = zscore(node, within_strength)
        rows.append(
            CommunityNodeMetrics(
                index,
                node,
                connectome.roi_labels[index],
                communities.partition[node],
                binary_participation,
                "available" if binary_reason is None else "unavailable",
                binary_reason,
                weighted_participation,
                "available" if weighted_reason is None else "unavailable",
                weighted_reason,
                within_degree[node],
                within_strength[node],
                degree_z,
                "available" if degree_z_reason is None else "unavailable",
                degree_z_reason,
                strength_z,
                "available" if strength_z_reason is None else "unavailable",
                strength_z_reason,
            )
        )
    return tuple(rows)


def write_community_outputs(
    communities: CommunityResults,
    output_dir: str | Path,
    *,
    node_metrics: tuple[CommunityNodeMetrics, ...] | None = None,
    hub_results: Any | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write community, optional participation, and optional hub outputs safely."""
    directory = Path(output_dir).resolve()
    paths = {
        "community_assignments": directory / "community_assignments.tsv",
        "community_summary": directory / "community_summary.json",
    }
    if node_metrics is not None:
        paths["participation_metrics"] = directory / "participation_metrics.tsv"
    if hub_results is not None:
        paths["hub_candidates"] = directory / "hub_candidates.tsv"
        paths["hub_summary"] = directory / "hub_summary.json"
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise CommunityOutputError(f"Refusing to overwrite existing community output(s): {names}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row.to_dict() for row in communities.assignments]).to_csv(
            paths["community_assignments"], sep="\t", index=False
        )
        paths["community_summary"].write_text(
            json.dumps(communities.summary_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if node_metrics is not None:
            pd.DataFrame([row.to_dict() for row in node_metrics]).to_csv(
                paths["participation_metrics"],
                sep="\t",
                index=False,
                float_format="%.17g",
            )
        if hub_results is not None:
            pd.DataFrame([row.to_dict() for row in hub_results.rows]).to_csv(
                paths["hub_candidates"], sep="\t", index=False, float_format="%.17g"
            )
            paths["hub_summary"].write_text(
                json.dumps(hub_results.summary_dict(), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError) as exc:
        raise CommunityOutputError(
            f"Could not write community outputs to '{directory}': {exc}"
        ) from exc
    return paths

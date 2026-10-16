"""Deterministic nodal metrics for binary and positive-weighted graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx

from brain_connectome_pipeline.config import TopologyConfig

MetricStatus = Literal["available", "unavailable"]


@dataclass(frozen=True)
class NodalMetricResult:
    roi_index: int
    roi_id: str
    label: str | None
    representation: Literal["binary", "weighted"]
    metric: str
    value: float | int | None
    status: MetricStatus
    definition: str
    edge_interpretation: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_index": self.roi_index,
            "roi_id": self.roi_id,
            "label": self.label,
            "representation": self.representation,
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
            "definition": self.definition,
            "edge_interpretation": self.edge_interpretation,
            "reason": self.reason,
        }


def _local_efficiency(graph: nx.Graph, node: str) -> float:
    neighbors = list(graph.neighbors(node))
    if len(neighbors) < 2:
        return 0.0
    return float(nx.global_efficiency(graph.subgraph(neighbors)))


def _safe_value(value: float | int) -> float | int | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compute_nodal_metrics(
    binary_graph: nx.Graph,
    weighted_graph: nx.Graph,
    roi_ids: tuple[str, ...],
    roi_labels: tuple[str | None, ...],
    config: TopologyConfig,
) -> tuple[NodalMetricResult, ...]:
    """Calculate nodal metrics without modifying either graph."""
    if tuple(binary_graph.nodes) != roi_ids or tuple(weighted_graph.nodes) != roi_ids:
        raise ValueError("Graph node order must exactly match the supplied ROI order.")
    if len(roi_labels) != len(roi_ids):
        raise ValueError("ROI labels must align exactly with ROI identifiers.")

    node_count = len(roi_ids)
    degree = dict(binary_graph.degree())
    degree_centrality = nx.degree_centrality(binary_graph) if node_count >= 2 else {}
    strength = dict(weighted_graph.degree(weight="weight"))
    binary_clustering = nx.clustering(binary_graph)
    weighted_clustering = nx.clustering(weighted_graph, weight="weight")
    binary_betweenness = nx.betweenness_centrality(binary_graph, normalized=True, weight=None)
    weighted_betweenness = nx.betweenness_centrality(
        weighted_graph, normalized=True, weight="distance"
    )
    binary_closeness = nx.closeness_centrality(binary_graph, distance=None, wf_improved=True)
    weighted_closeness = nx.closeness_centrality(
        weighted_graph, distance="distance", wf_improved=True
    )
    local_efficiency = {node: _local_efficiency(binary_graph, node) for node in roi_ids}

    eigenvector: dict[str, float] | None = None
    eigenvector_reason: str | None = None
    if node_count == 0:
        eigenvector_reason = "Eigenvector centrality is undefined for an empty graph."
    elif not nx.is_connected(weighted_graph):
        eigenvector_reason = (
            "Weighted graph is disconnected; a unique whole-graph eigenvector ranking is not "
            "reported."
        )
    elif node_count == 1:
        eigenvector = {roi_ids[0]: 1.0}
    else:
        try:
            candidate = nx.eigenvector_centrality(
                weighted_graph,
                max_iter=config.eigenvector_max_iterations,
                tol=config.eigenvector_tolerance,
                weight="weight",
            )
            if not all(math.isfinite(value) for value in candidate.values()):
                eigenvector_reason = "Eigenvector solver returned a nonfinite value."
            else:
                eigenvector = candidate
        except nx.PowerIterationFailedConvergence:
            eigenvector_reason = (
                "Weighted eigenvector centrality did not converge within the configured limit."
            )
        except (nx.NetworkXException, ArithmeticError) as exc:
            eigenvector_reason = f"Weighted eigenvector centrality is unavailable: {exc}"

    results: list[NodalMetricResult] = []

    def add(
        index: int,
        representation: Literal["binary", "weighted"],
        metric: str,
        value: float | int | None,
        definition: str,
        edge_interpretation: str,
        reason: str | None = None,
    ) -> None:
        safe = None if value is None else _safe_value(value)
        if safe is None:
            results.append(
                NodalMetricResult(
                    index,
                    roi_ids[index],
                    roi_labels[index],
                    representation,
                    metric,
                    None,
                    "unavailable",
                    definition,
                    edge_interpretation,
                    reason or "Metric produced a nonfinite value.",
                )
            )
        else:
            results.append(
                NodalMetricResult(
                    index,
                    roi_ids[index],
                    roi_labels[index],
                    representation,
                    metric,
                    safe,
                    "available",
                    definition,
                    edge_interpretation,
                )
            )

    for index, node in enumerate(roi_ids):
        add(index, "binary", "degree", int(degree[node]), "Number of incident edges.", "edge")
        add(
            index,
            "binary",
            "degree_centrality",
            degree_centrality.get(node),
            "Degree divided by N-1.",
            "edge",
            "Degree centrality is undefined when N is less than 2.",
        )
        add(
            index,
            "weighted",
            "strength",
            float(strength[node]),
            "Sum of incident retained positive strengths.",
            "raw positive strength",
        )
        add(
            index,
            "binary",
            "local_clustering",
            float(binary_clustering[node]),
            "Fraction of possible neighbor-neighbor edges that are present.",
            "edge",
        )
        add(
            index,
            "weighted",
            "local_clustering",
            float(weighted_clustering[node]),
            "NetworkX geometric-mean weighted clustering normalized by maximum graph weight.",
            "raw positive strength normalized internally by NetworkX",
        )
        add(
            index,
            "binary",
            "betweenness_centrality",
            float(binary_betweenness[node]),
            "Normalized fraction of binary shortest paths passing through the node.",
            "unit-hop distance",
        )
        add(
            index,
            "weighted",
            "betweenness_centrality",
            float(weighted_betweenness[node]),
            "Normalized fraction of reciprocal-strength shortest paths passing through the node.",
            "distance = 1 / positive strength",
        )
        add(
            index,
            "binary",
            "closeness_centrality",
            float(binary_closeness[node]),
            "Reciprocal mean binary distance with Wasserman-Faust disconnected-graph correction.",
            "unit-hop distance",
        )
        add(
            index,
            "weighted",
            "closeness_centrality",
            float(weighted_closeness[node]),
            "Reciprocal mean weighted distance with Wasserman-Faust correction.",
            "distance = 1 / positive strength",
        )
        add(
            index,
            "binary",
            "local_efficiency",
            local_efficiency[node],
            "Binary global efficiency of the subgraph induced by immediate neighbors.",
            "unit-hop distance among neighbors",
        )
        add(
            index,
            "weighted",
            "eigenvector_centrality",
            None if eigenvector is None else float(eigenvector[node]),
            "Positive-strength eigenvector centrality from the dominant adjacency eigenvector.",
            "raw positive strength",
            eigenvector_reason,
        )
    return tuple(results)

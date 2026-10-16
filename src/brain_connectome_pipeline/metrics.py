"""Global and nodal topology metrics with explicit availability metadata."""

from __future__ import annotations

import json
import math
import warnings as python_warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import pandas as pd

from brain_connectome_pipeline.centrality import NodalMetricResult, compute_nodal_metrics
from brain_connectome_pipeline.config import GraphConfig, TopologyConfig
from brain_connectome_pipeline.graph import ConstructedGraphs
from brain_connectome_pipeline.models import ConnectomeData


class TopologyAnalysisError(ValueError):
    """Raised when topology analysis cannot safely operate on the supplied graphs."""


class TopologyOutputError(OSError):
    """Raised when topology outputs cannot be written safely."""


@dataclass(frozen=True)
class GlobalMetricResult:
    metric: str
    representation: Literal["binary", "weighted"]
    value: float | int | None
    status: Literal["available", "unavailable"]
    definition: str
    edge_interpretation: str
    scope: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "representation": self.representation,
            "value": self.value,
            "status": self.status,
            "definition": self.definition,
            "edge_interpretation": self.edge_interpretation,
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TopologyResults:
    global_metrics: tuple[GlobalMetricResult, ...]
    nodal_metrics: tuple[NodalMetricResult, ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "global_metrics": [metric.to_dict() for metric in self.global_metrics],
            "nodal_metrics": [metric.to_dict() for metric in self.nodal_metrics],
            "warnings": list(self.warnings),
        }

    def summary_dict(self) -> dict[str, Any]:
        global_available = sum(metric.status == "available" for metric in self.global_metrics)
        nodal_available = sum(metric.status == "available" for metric in self.nodal_metrics)
        return {
            "global_metrics": len(self.global_metrics),
            "global_metrics_available": global_available,
            "global_metrics_unavailable": len(self.global_metrics) - global_available,
            "nodal_metric_rows": len(self.nodal_metrics),
            "nodal_metrics_available": nodal_available,
            "nodal_metrics_unavailable": len(self.nodal_metrics) - nodal_available,
            "warnings": list(self.warnings),
            "analysis_metadata": self.metadata,
        }


def _validate_graphs(graphs: ConstructedGraphs, roi_ids: tuple[str, ...]) -> None:
    if tuple(graphs.binary_graph.nodes) != roi_ids or tuple(graphs.weighted_graph.nodes) != roi_ids:
        raise TopologyAnalysisError("Graph node order does not match the original ROI order.")
    binary_edges = {frozenset(edge) for edge in graphs.binary_graph.edges}
    weighted_edges = {frozenset(edge) for edge in graphs.weighted_graph.edges}
    if binary_edges != weighted_edges:
        raise TopologyAnalysisError("Binary and weighted graphs must contain the same edge set.")
    for source, target, data in graphs.weighted_graph.edges(data=True):
        weight = data.get("weight")
        distance = data.get("distance")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise TopologyAnalysisError(
                f"Weighted edge {source!r}-{target!r} lacks a finite positive strength."
            )
        if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance <= 0:
            raise TopologyAnalysisError(
                f"Weighted edge {source!r}-{target!r} lacks a finite positive distance."
            )


def _weighted_global_efficiency(graph: nx.Graph) -> float | None:
    node_count = graph.number_of_nodes()
    if node_count < 2:
        return None
    pair_count = node_count * (node_count - 1)
    efficiency = 0.0
    for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight="distance"):
        for target, distance in lengths.items():
            if source != target:
                efficiency += (1.0 / distance) / pair_count
    return efficiency if math.isfinite(efficiency) else None


def _largest_component(graph: nx.Graph, node_order: tuple[str, ...]) -> tuple[str, ...]:
    if graph.number_of_nodes() == 0:
        return ()
    order = {node: index for index, node in enumerate(node_order)}
    components = list(nx.connected_components(graph))
    chosen = min(components, key=lambda nodes: (-len(nodes), min(order[node] for node in nodes)))
    return tuple(node for node in node_order if node in chosen)


def compute_global_metrics(
    graphs: ConstructedGraphs,
    config: TopologyConfig,
    roi_ids: tuple[str, ...],
) -> tuple[GlobalMetricResult, ...]:
    """Calculate global metrics using explicit binary and weighted edge interpretations."""
    binary = graphs.binary_graph
    weighted = graphs.weighted_graph
    node_count = binary.number_of_nodes()
    edge_count = binary.number_of_edges()
    components = nx.number_connected_components(binary) if node_count else 0
    isolates = nx.number_of_isolates(binary)
    connected = node_count > 0 and nx.is_connected(binary)
    largest_nodes = _largest_component(binary, roi_ids)
    largest_count = len(largest_nodes)
    largest_fraction = largest_count / node_count if node_count else None
    results: list[GlobalMetricResult] = []

    def add(
        metric: str,
        representation: Literal["binary", "weighted"],
        value: float | int | None,
        definition: str,
        edge_interpretation: str,
        scope: str = "whole_graph",
        reason: str | None = None,
    ) -> None:
        if value is not None and isinstance(value, float) and not math.isfinite(value):
            value = None
            reason = reason or "Metric produced a nonfinite value."
        results.append(
            GlobalMetricResult(
                metric,
                representation,
                value,
                "available" if value is not None else "unavailable",
                definition,
                edge_interpretation,
                scope,
                None if value is not None else reason or "Metric is mathematically undefined.",
            )
        )

    add("number_of_nodes", "binary", node_count, "Number of graph nodes.", "edge")
    add("number_of_edges", "binary", edge_count, "Number of retained undirected edges.", "edge")
    add(
        "edge_density",
        "binary",
        float(nx.density(binary)),
        "Retained edges divided by N(N-1)/2; zero by convention when fewer than two nodes.",
        "edge",
    )
    add(
        "connected_components",
        "binary",
        components,
        "Number of connected components, including isolated nodes.",
        "unit-hop connectivity",
    )
    add("isolated_nodes", "binary", isolates, "Number of degree-zero nodes.", "edge")
    add(
        "largest_component_fraction",
        "binary",
        largest_fraction,
        "Largest connected-component node count divided by N.",
        "unit-hop connectivity",
        reason="Largest-component fraction is undefined for an empty graph.",
    )
    add(
        "mean_clustering_coefficient",
        "binary",
        float(nx.average_clustering(binary)) if node_count else None,
        "Mean NetworkX binary local clustering coefficient across nodes.",
        "edge",
        reason="Mean clustering is undefined for an empty graph.",
    )
    add(
        "transitivity",
        "binary",
        float(nx.transitivity(binary)),
        "Three times triangles divided by connected triples; NetworkX returns zero "
        "with no triples.",
        "edge",
    )
    add(
        "global_efficiency",
        "binary",
        float(nx.global_efficiency(binary)) if node_count >= 2 else None,
        "Mean reciprocal binary shortest-path distance over all ordered node pairs; "
        "unreachable pairs contribute zero.",
        "unit-hop distance",
        reason="Global efficiency is undefined when fewer than two nodes are present.",
    )
    whole_binary_path = (
        float(nx.average_shortest_path_length(binary)) if connected and node_count >= 2 else None
    )
    add(
        "characteristic_path_length",
        "binary",
        whole_binary_path,
        "Mean shortest unit-hop distance over all node pairs in a connected graph.",
        "unit-hop distance",
        reason=("Whole-graph path length requires a connected graph with at least two nodes."),
    )
    if config.report_largest_component_paths:
        component_scope = (
            f"largest_component(nodes={largest_count}, coverage={largest_fraction:.6g})"
            if largest_fraction is not None
            else "largest_component(nodes=0, coverage=undefined)"
        )
        component_binary_path = (
            float(nx.average_shortest_path_length(binary.subgraph(largest_nodes)))
            if largest_count >= 2
            else None
        )
        add(
            "characteristic_path_length_largest_component",
            "binary",
            component_binary_path,
            "Mean unit-hop path length within the largest connected component.",
            "unit-hop distance",
            component_scope,
            "Largest component contains fewer than two nodes.",
        )
    with python_warnings.catch_warnings():
        python_warnings.simplefilter("ignore", RuntimeWarning)
        try:
            assortativity = float(nx.degree_assortativity_coefficient(binary))
        except (nx.NetworkXException, ZeroDivisionError):
            assortativity = math.nan
    add(
        "degree_assortativity",
        "binary",
        assortativity,
        "Pearson correlation of degrees at the two ends of retained edges.",
        "endpoint degree",
        reason=(
            "Degree assortativity is undefined because endpoint degree variance is zero or absent."
        ),
    )

    add(
        "mean_clustering_coefficient",
        "weighted",
        float(nx.average_clustering(weighted, weight="weight")) if node_count else None,
        "Mean NetworkX weighted clustering using geometric means after normalization "
        "by maximum edge weight.",
        "raw positive strength normalized internally by NetworkX",
        reason="Mean weighted clustering is undefined for an empty graph.",
    )
    total_strength = float(weighted.size(weight="weight"))
    add(
        "mean_node_strength",
        "weighted",
        (2.0 * (total_strength / node_count)) if node_count else None,
        "Mean sum of incident retained positive strengths across nodes.",
        "raw positive strength",
        reason="Mean node strength is undefined for an empty graph.",
    )
    add(
        "total_retained_positive_strength",
        "weighted",
        total_strength,
        "Sum of each retained undirected positive edge strength once.",
        "raw positive strength",
    )
    add(
        "global_efficiency",
        "weighted",
        _weighted_global_efficiency(weighted),
        "Mean inverse reciprocal-strength shortest-path distance over ordered pairs; "
        "unreachable pairs contribute zero.",
        "distance = 1 / positive strength",
        reason="Weighted global efficiency is undefined when fewer than two nodes are present.",
    )
    whole_weighted_path = (
        float(nx.average_shortest_path_length(weighted, weight="distance", method="dijkstra"))
        if connected and node_count >= 2
        else None
    )
    add(
        "characteristic_path_length",
        "weighted",
        whole_weighted_path,
        "Mean reciprocal-strength shortest-path cost over all pairs in a connected graph.",
        "distance = 1 / positive strength",
        reason=(
            "Whole-graph weighted path length requires a connected graph with at least two nodes."
        ),
    )
    if config.report_largest_component_paths:
        component_scope = (
            f"largest_component(nodes={largest_count}, coverage={largest_fraction:.6g})"
            if largest_fraction is not None
            else "largest_component(nodes=0, coverage=undefined)"
        )
        component_weighted_path = (
            float(
                nx.average_shortest_path_length(
                    weighted.subgraph(largest_nodes), weight="distance", method="dijkstra"
                )
            )
            if largest_count >= 2
            else None
        )
        add(
            "characteristic_path_length_largest_component",
            "weighted",
            component_weighted_path,
            "Mean reciprocal-strength path cost within the largest connected component.",
            "distance = 1 / positive strength",
            component_scope,
            "Largest component contains fewer than two nodes.",
        )
    return tuple(results)


def analyze_topology(
    graphs: ConstructedGraphs,
    connectome: ConnectomeData,
    topology_config: TopologyConfig,
    graph_config: GraphConfig,
) -> TopologyResults:
    """Calculate global and nodal topology with JSON-safe availability reporting."""
    if not topology_config.enabled:
        raise TopologyAnalysisError("Topology analysis is disabled in the configuration.")
    _validate_graphs(graphs, connectome.roi_ids)
    global_metrics = compute_global_metrics(graphs, topology_config, connectome.roi_ids)
    try:
        nodal_metrics = compute_nodal_metrics(
            graphs.binary_graph,
            graphs.weighted_graph,
            connectome.roi_ids,
            connectome.roi_labels,
            topology_config,
        )
    except ValueError as exc:
        raise TopologyAnalysisError(str(exc)) from exc
    warning_values: list[str] = []
    for metric in (*global_metrics, *nodal_metrics):
        if metric.status == "unavailable" and metric.reason is not None:
            message = f"{metric.metric}: {metric.reason}"
            if message not in warning_values:
                warning_values.append(message)
    metadata = {
        "number_of_rois": connectome.number_of_rois,
        "roi_order": list(connectome.roi_ids),
        "matrix_semantics": connectome.matrix_semantics,
        "threshold_mode": graphs.diagnostics.threshold_mode,
        "requested_density": graphs.diagnostics.requested_density,
        "achieved_density": graphs.diagnostics.achieved_density,
        "retained_edges": graphs.diagnostics.retained_edges,
        "positive_edge_policy": graph_config.positive_edge_policy,
        "weighted_distance": "1 / positive strength",
        "binary_distance": "unit hop",
        "closeness_disconnected_normalization": "Wasserman-Faust correction",
        "largest_component_paths_reported": topology_config.report_largest_component_paths,
        "eigenvector_max_iterations": topology_config.eigenvector_max_iterations,
        "eigenvector_tolerance": topology_config.eigenvector_tolerance,
    }
    return TopologyResults(global_metrics, nodal_metrics, metadata, tuple(warning_values))


def write_topology_outputs(
    results: TopologyResults,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write deterministic topology tables after an all-file collision preflight."""
    directory = Path(output_dir).resolve()
    paths = {
        "global_metrics": directory / "global_metrics.tsv",
        "nodal_metrics": directory / "nodal_metrics.tsv",
        "topology_results": directory / "topology_results.json",
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise TopologyOutputError(f"Refusing to overwrite existing topology output(s): {names}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metric.to_dict() for metric in results.global_metrics]).to_csv(
            paths["global_metrics"], sep="\t", index=False, float_format="%.17g"
        )
        pd.DataFrame([metric.to_dict() for metric in results.nodal_metrics]).to_csv(
            paths["nodal_metrics"], sep="\t", index=False, float_format="%.17g"
        )
        paths["topology_results"].write_text(
            json.dumps(results.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        raise TopologyOutputError(
            f"Could not write topology outputs to '{directory}': {exc}"
        ) from exc
    return paths

"""Positive undirected graph construction and deterministic graph outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from brain_connectome_pipeline.config import GraphConfig
from brain_connectome_pipeline.models import ConnectomeData


class GraphConstructionError(ValueError):
    """Raised when graph settings or eligible edge values are invalid."""


class GraphOutputError(OSError):
    """Raised when graph outputs cannot be written safely."""


@dataclass(frozen=True)
class SelectedEdge:
    source_index: int
    target_index: int
    weight: float


@dataclass(frozen=True)
class GraphDiagnostics:
    number_of_nodes: int
    number_of_possible_edges: int
    positive_eligible_edges: int
    negative_excluded_edges: int
    zero_entries: int
    retained_edges: int
    target_edge_count: int | None
    requested_density: float | None
    achieved_density: float
    threshold_mode: str
    minimum_retained_weight: float | None
    median_retained_weight: float | None
    maximum_retained_weight: float | None
    isolated_nodes: int
    connected_components: int
    largest_connected_component_size: int
    connected: bool
    edge_weight_tie_count: int
    boundary_tie_count: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number_of_nodes": self.number_of_nodes,
            "number_of_possible_edges": self.number_of_possible_edges,
            "positive_eligible_edges": self.positive_eligible_edges,
            "negative_excluded_edges": self.negative_excluded_edges,
            "zero_entries": self.zero_entries,
            "retained_edges": self.retained_edges,
            "target_edge_count": self.target_edge_count,
            "requested_density": self.requested_density,
            "achieved_density": self.achieved_density,
            "threshold_mode": self.threshold_mode,
            "minimum_retained_weight": self.minimum_retained_weight,
            "median_retained_weight": self.median_retained_weight,
            "maximum_retained_weight": self.maximum_retained_weight,
            "isolated_nodes": self.isolated_nodes,
            "connected_components": self.connected_components,
            "largest_connected_component_size": self.largest_connected_component_size,
            "connected": self.connected,
            "edge_weight_tie_count": self.edge_weight_tie_count,
            "boundary_tie_count": self.boundary_tie_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ConstructedGraphs:
    binary_graph: nx.Graph
    weighted_graph: nx.Graph
    selected_edges: tuple[SelectedEdge, ...]
    diagnostics: GraphDiagnostics

    def summary_dict(self) -> dict[str, Any]:
        return self.diagnostics.to_dict()


def _node_attributes(connectome: ConnectomeData, index: int) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "roi_index": index,
        "roi_id": connectome.roi_ids[index],
    }
    label = connectome.roi_labels[index]
    if label is not None:
        attributes["label"] = label
    if connectome.atlas is not None:
        attributes["atlas"] = connectome.atlas
    coordinates = connectome.coordinates[index]
    if coordinates is not None:
        attributes["coordinates"] = coordinates
        attributes["x"], attributes["y"], attributes["z"] = coordinates
    return attributes


def _eligible_edges(connectome: ConnectomeData) -> list[SelectedEdge]:
    matrix = connectome.matrix
    if not np.isfinite(matrix).all():
        raise GraphConstructionError("Graph construction requires finite matrix weights.")
    edges: list[SelectedEdge] = []
    for source in range(connectome.number_of_rois):
        for target in range(source + 1, connectome.number_of_rois):
            value = float(matrix[source, target])
            if value <= 0:
                continue
            edges.append(SelectedEdge(source, target, value))
    return edges


def _half_up_edge_count(density: float, possible_edges: int) -> int:
    """Convert density to edges using floor(density * possible + 0.5)."""
    return math.floor(density * possible_edges + 0.5)


def _tie_diagnostics(eligible: list[SelectedEdge], retained_count: int) -> tuple[int, int]:
    counts: dict[float, int] = {}
    for edge in eligible:
        counts[edge.weight] = counts.get(edge.weight, 0) + 1
    tied_edges = sum(count for count in counts.values() if count > 1)
    boundary_ties = 0
    if 0 < retained_count < len(eligible):
        boundary_weight = eligible[retained_count - 1].weight
        if eligible[retained_count].weight == boundary_weight:
            boundary_ties = counts[boundary_weight]
    return tied_edges, boundary_ties


def construct_graphs(connectome: ConnectomeData, graph_config: GraphConfig) -> ConstructedGraphs:
    """Build binary and weighted positive graphs from one deterministic edge set."""
    if not graph_config.enabled:
        raise GraphConstructionError("Graph construction is disabled in the configuration.")
    if graph_config.positive_edge_policy != "strictly_positive":
        raise GraphConstructionError("Only the 'strictly_positive' edge policy is supported.")
    if graph_config.tie_handling != "roi_index":
        raise GraphConstructionError("Only deterministic 'roi_index' tie handling is supported.")

    number_of_nodes = connectome.number_of_rois
    possible_edges = number_of_nodes * (number_of_nodes - 1) // 2
    eligible = _eligible_edges(connectome)
    eligible.sort(key=lambda edge: (-edge.weight, edge.source_index, edge.target_index))
    requested_density: float | None = None
    target_count: int | None = None
    threshold_mode = graph_config.mode
    warnings: list[str] = []

    if graph_config.mode == "unthresholded":
        retained_count = len(eligible)
    elif graph_config.mode == "proportional":
        if graph_config.target_density is None or not 0 <= graph_config.target_density <= 1:
            raise GraphConstructionError("Proportional mode requires density between 0 and 1.")
        requested_density = graph_config.target_density
        target_count = _half_up_edge_count(requested_density, possible_edges)
        if (
            connectome.matrix_semantics == "binary_adjacency"
            and not graph_config.allow_binary_subsampling
        ):
            retained_count = len(eligible)
            threshold_mode = "binary_topology_preserved"
            warnings.append(
                "Binary topology was preserved; proportional subsampling requires explicit opt-in."
            )
            if retained_count < target_count:
                warnings.append(
                    "Insufficient positive eligible edges prevented reaching the requested density."
                )
        else:
            retained_count = min(target_count, len(eligible))
            if connectome.matrix_semantics == "binary_adjacency" and target_count < len(eligible):
                warnings.append(
                    "Binary equal-weight subsampling is index-dependent, "
                    "not physiologically ranked."
                )
            if retained_count < target_count:
                warnings.append(
                    "Insufficient positive eligible edges prevented reaching the requested density."
                )
    else:
        raise GraphConstructionError(f"Unsupported graph construction mode: {graph_config.mode}")

    tied_edges, boundary_ties = _tie_diagnostics(eligible, retained_count)
    if boundary_ties:
        warnings.append(
            "The retention boundary split an exact-weight tie; ROI index order selected the edges."
        )
    selected = tuple(eligible[:retained_count])

    binary_graph = nx.Graph(representation="binary", matrix_semantics=connectome.matrix_semantics)
    weighted_graph = nx.Graph(
        representation="weighted_strength", matrix_semantics=connectome.matrix_semantics
    )
    for index, roi_id in enumerate(connectome.roi_ids):
        attributes = _node_attributes(connectome, index)
        binary_graph.add_node(roi_id, **attributes)
        weighted_graph.add_node(roi_id, **attributes)
    for edge in selected:
        source = connectome.roi_ids[edge.source_index]
        target = connectome.roi_ids[edge.target_index]
        try:
            distance = 1.0 / edge.weight
        except OverflowError as exc:
            raise GraphConstructionError(
                f"Reciprocal-strength distance overflowed for edge {source!r}-{target!r}."
            ) from exc
        if not math.isfinite(distance) or distance <= 0:
            raise GraphConstructionError(
                f"Edge {source!r}-{target!r} cannot produce a finite positive distance."
            )
        binary_graph.add_edge(source, target)
        weighted_graph.add_edge(
            source,
            target,
            weight=edge.weight,
            distance=distance,
        )

    achieved_density = len(selected) / possible_edges if possible_edges else 0.0
    if requested_density is not None and achieved_density < requested_density:
        warnings.append("Achieved density is below the requested density.")
    isolated_nodes = nx.number_of_isolates(binary_graph)
    components = list(nx.connected_components(binary_graph))
    component_count = len(components)
    largest_component = max((len(component) for component in components), default=0)
    connected = nx.is_connected(binary_graph)
    retained_weights = np.asarray([edge.weight for edge in selected], dtype=np.float64)
    if retained_weights.size:
        minimum_weight = float(np.min(retained_weights))
        median_weight = float(np.median(retained_weights))
        maximum_weight = float(np.max(retained_weights))
    else:
        minimum_weight = median_weight = maximum_weight = None
    if achieved_density < 0.10:
        warnings.append("Constructed graph density is below 0.10 and the graph is sparse.")
    if not connected:
        warnings.append(
            f"Constructed graph is disconnected ({component_count} connected components, "
            f"{isolated_nodes} isolated nodes)."
        )

    diagnostics = GraphDiagnostics(
        number_of_nodes=number_of_nodes,
        number_of_possible_edges=possible_edges,
        positive_eligible_edges=len(eligible),
        negative_excluded_edges=connectome.diagnostics.negative_edges,
        zero_entries=connectome.diagnostics.zero_edges,
        retained_edges=len(selected),
        target_edge_count=target_count,
        requested_density=requested_density,
        achieved_density=achieved_density,
        threshold_mode=threshold_mode,
        minimum_retained_weight=minimum_weight,
        median_retained_weight=median_weight,
        maximum_retained_weight=maximum_weight,
        isolated_nodes=isolated_nodes,
        connected_components=component_count,
        largest_connected_component_size=largest_component,
        connected=connected,
        edge_weight_tie_count=tied_edges,
        boundary_tie_count=boundary_ties,
        warnings=tuple(warnings),
    )
    return ConstructedGraphs(binary_graph, weighted_graph, selected, diagnostics)


def _adjacency(graph: nx.Graph, roi_ids: tuple[str, ...], *, weighted: bool) -> pd.DataFrame:
    if weighted:
        values = nx.to_numpy_array(graph, nodelist=roi_ids, weight="weight", dtype=np.float64)
    else:
        values = nx.to_numpy_array(graph, nodelist=roi_ids, weight=None, dtype=np.int8)
    return pd.DataFrame(values, index=roi_ids, columns=roi_ids)


def write_graph_outputs(
    graphs: ConstructedGraphs,
    connectome: ConnectomeData,
    output_dir: str | Path,
    *,
    write_binary: bool = True,
    write_weighted: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write deterministic graph artifacts after an all-file collision preflight."""
    directory = Path(output_dir).resolve()
    filenames = {
        "edge_list": "edges.tsv",
        "node_metadata": "nodes.tsv",
        "diagnostics": "graph_diagnostics.json",
    }
    if write_binary:
        filenames["binary_adjacency"] = "binary_adjacency.tsv"
    if write_weighted:
        filenames["weighted_adjacency"] = "weighted_adjacency.tsv"
    paths = {name: directory / filename for name, filename in filenames.items()}
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise GraphOutputError(f"Refusing to overwrite existing graph output(s): {names}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if write_binary:
            _adjacency(graphs.binary_graph, connectome.roi_ids, weighted=False).to_csv(
                paths["binary_adjacency"], sep="\t", index_label="roi_id"
            )
        if write_weighted:
            _adjacency(graphs.weighted_graph, connectome.roi_ids, weighted=True).to_csv(
                paths["weighted_adjacency"],
                sep="\t",
                index_label="roi_id",
                float_format="%.17g",
            )
        edge_rows = [
            {
                "source_index": edge.source_index,
                "target_index": edge.target_index,
                "source": connectome.roi_ids[edge.source_index],
                "target": connectome.roi_ids[edge.target_index],
                "weight": edge.weight,
                "distance": 1.0 / edge.weight,
            }
            for edge in sorted(
                graphs.selected_edges, key=lambda value: (value.source_index, value.target_index)
            )
        ]
        pd.DataFrame(
            edge_rows,
            columns=["source_index", "target_index", "source", "target", "weight", "distance"],
        ).to_csv(paths["edge_list"], sep="\t", index=False, float_format="%.17g")
        node_rows = []
        for index, roi_id in enumerate(connectome.roi_ids):
            coordinates = connectome.coordinates[index]
            node_rows.append(
                {
                    "roi_index": index,
                    "roi_id": roi_id,
                    "label": connectome.roi_labels[index],
                    "atlas": connectome.atlas,
                    "x": None if coordinates is None else coordinates[0],
                    "y": None if coordinates is None else coordinates[1],
                    "z": None if coordinates is None else coordinates[2],
                }
            )
        pd.DataFrame(node_rows).to_csv(paths["node_metadata"], sep="\t", index=False)
        paths["diagnostics"].write_text(
            json.dumps(graphs.diagnostics.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        raise GraphOutputError(f"Could not write graph outputs to '{directory}': {exc}") from exc
    return paths

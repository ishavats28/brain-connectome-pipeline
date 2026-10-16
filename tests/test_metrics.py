from __future__ import annotations

import copy
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.config import GraphConfig, TopologyConfig
from brain_connectome_pipeline.graph import ConstructedGraphs, GraphDiagnostics, construct_graphs
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.metrics import (
    TopologyAnalysisError,
    TopologyOutputError,
    analyze_topology,
    compute_global_metrics,
    write_topology_outputs,
)
from brain_connectome_pipeline.models import ConnectomeData


def _analysis(matrix: np.ndarray, topology: TopologyConfig | None = None):
    count = matrix.shape[0]
    connectome = ConnectomeData(
        matrix=matrix,
        roi_ids=tuple(f"R{index}" for index in range(count)),
        roi_labels=tuple(f"Region {index}" for index in range(count)),
        coordinates=(None,) * count,
        atlas=None,
        input_type="matrix",
        matrix_semantics="correlation",
        source_path=Path("synthetic.npy"),
        label_source="test",
        diagnostics=validate_matrix(matrix, "correlation", 1e-8, 1e-8),
    )
    graph_config = GraphConfig(enabled=True)
    graphs = construct_graphs(connectome, graph_config)
    results = analyze_topology(
        graphs,
        connectome,
        topology or TopologyConfig(enabled=True),
        graph_config,
    )
    return connectome, graphs, results


def _global(results, metric: str, representation: str, scope: str = "whole_graph"):
    return next(
        row
        for row in results.global_metrics
        if row.metric == metric and row.representation == representation and row.scope == scope
    )


def _nodal(results, roi_id: str, metric: str, representation: str):
    return next(
        row
        for row in results.nodal_metrics
        if row.roi_id == roi_id and row.metric == metric and row.representation == representation
    )


def test_triangle_global_binary_and_uniform_weight_metrics() -> None:
    matrix = np.full((3, 3), 0.5)
    np.fill_diagonal(matrix, 1.0)
    _, _, results = _analysis(matrix)
    assert _global(results, "number_of_nodes", "binary").value == 3
    assert _global(results, "number_of_edges", "binary").value == 3
    assert _global(results, "edge_density", "binary").value == 1.0
    assert _global(results, "mean_clustering_coefficient", "binary").value == 1.0
    assert _global(results, "transitivity", "binary").value == 1.0
    assert _global(results, "global_efficiency", "binary").value == 1.0
    assert _global(results, "characteristic_path_length", "binary").value == 1.0
    assert _global(results, "mean_clustering_coefficient", "weighted").value == 1.0
    assert _global(results, "mean_node_strength", "weighted").value == 1.0
    assert _global(results, "total_retained_positive_strength", "weighted").value == 1.5
    assert _global(results, "characteristic_path_length", "weighted").value == 2.0
    assert _global(results, "global_efficiency", "weighted").value == pytest.approx(0.5)
    assert _nodal(results, "R0", "local_clustering", "binary").value == 1.0
    assert _nodal(results, "R0", "local_clustering", "weighted").value == 1.0
    assert _nodal(results, "R0", "local_efficiency", "binary").value == 1.0
    assert _nodal(results, "R0", "eigenvector_centrality", "weighted").value == pytest.approx(
        1 / np.sqrt(3)
    )
    assortativity = _global(results, "degree_assortativity", "binary")
    assert assortativity.value is None
    assert assortativity.status == "unavailable"
    assert "variance" in assortativity.reason


def test_path_graph_efficiency_and_exact_path_length() -> None:
    matrix = np.eye(4)
    for source, target in ((0, 1), (1, 2), (2, 3)):
        matrix[source, target] = matrix[target, source] = 1.0
    _, _, results = _analysis(matrix)
    assert _global(results, "edge_density", "binary").value == 0.5
    assert _global(results, "global_efficiency", "binary").value == pytest.approx(13 / 18)
    assert _global(results, "characteristic_path_length", "binary").value == pytest.approx(5 / 3)
    assert _global(results, "mean_clustering_coefficient", "binary").value == 0.0
    assert _global(results, "transitivity", "binary").value == 0.0


def test_disconnected_paths_components_closeness_and_eigenvector_unavailability() -> None:
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 1.0
    matrix[2, 3] = matrix[3, 2] = 1.0
    _, _, results = _analysis(matrix)
    assert _global(results, "connected_components", "binary").value == 2
    assert _global(results, "global_efficiency", "binary").value == pytest.approx(1 / 3)
    whole_path = _global(results, "characteristic_path_length", "binary")
    assert whole_path.value is None
    assert whole_path.status == "unavailable"
    largest = next(
        row
        for row in results.global_metrics
        if row.metric == "characteristic_path_length_largest_component"
        and row.representation == "binary"
    )
    assert largest.value == 1.0
    assert "nodes=2" in largest.scope
    assert "coverage=0.5" in largest.scope
    assert _nodal(results, "R0", "closeness_centrality", "binary").value == pytest.approx(1 / 3)
    for roi_id in ("R0", "R1", "R2", "R3"):
        eigenvector = _nodal(results, roi_id, "eigenvector_centrality", "weighted")
        assert eigenvector.value is None
        assert eigenvector.status == "unavailable"
        assert "disconnected" in eigenvector.reason


def test_star_nodal_degree_betweenness_strength_and_local_efficiency() -> None:
    matrix = np.eye(4)
    for leaf in (1, 2, 3):
        matrix[0, leaf] = matrix[leaf, 0] = 0.5
    _, _, results = _analysis(matrix)
    assert _nodal(results, "R0", "degree", "binary").value == 3
    assert _nodal(results, "R0", "degree_centrality", "binary").value == 1.0
    assert _nodal(results, "R0", "strength", "weighted").value == 1.5
    assert _nodal(results, "R0", "betweenness_centrality", "binary").value == 1.0
    assert _nodal(results, "R1", "betweenness_centrality", "binary").value == 0.0
    assert _nodal(results, "R0", "local_clustering", "binary").value == 0.0
    assert _nodal(results, "R0", "local_efficiency", "binary").value == 0.0
    assert _nodal(results, "R1", "closeness_centrality", "binary").value == 0.6


def test_weighted_distance_changes_paths_and_centrality() -> None:
    matrix = np.array([[1.0, 1.0, 0.1], [1.0, 1.0, 1.0], [0.1, 1.0, 1.0]])
    _, _, results = _analysis(matrix)
    expected_clustering = 0.1 ** (1 / 3)
    assert _global(results, "mean_clustering_coefficient", "weighted").value == pytest.approx(
        expected_clustering
    )
    assert _nodal(results, "R1", "betweenness_centrality", "binary").value == 0.0
    assert _nodal(results, "R1", "betweenness_centrality", "weighted").value == 1.0
    assert _global(results, "characteristic_path_length", "weighted").value == pytest.approx(4 / 3)
    assert _global(results, "global_efficiency", "weighted").value == pytest.approx(5 / 6)


def test_roi_order_graph_immutability_and_deterministic_execution() -> None:
    matrix = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.6], [0.0, 0.6, 1.0]])
    connectome, graphs, first = _analysis(matrix)
    binary_before = copy.deepcopy(list(graphs.binary_graph.edges(data=True)))
    weighted_before = copy.deepcopy(list(graphs.weighted_graph.edges(data=True)))
    second = analyze_topology(
        graphs, connectome, TopologyConfig(enabled=True), GraphConfig(enabled=True)
    )
    assert first.to_dict() == second.to_dict()
    assert list(graphs.binary_graph.edges(data=True)) == binary_before
    assert list(graphs.weighted_graph.edges(data=True)) == weighted_before
    observed_order = []
    for row in first.nodal_metrics:
        if row.roi_id not in observed_order:
            observed_order.append(row.roi_id)
    assert observed_order == list(connectome.roi_ids)


def test_eigenvector_nonconvergence_is_reported() -> None:
    matrix = np.eye(4)
    for source, target in ((0, 1), (1, 2), (2, 3)):
        matrix[source, target] = matrix[target, source] = 1.0
    topology = TopologyConfig(
        enabled=True, eigenvector_max_iterations=1, eigenvector_tolerance=1e-30
    )
    _, _, results = _analysis(matrix, topology)
    eigenvector_rows = [
        row for row in results.nodal_metrics if row.metric == "eigenvector_centrality"
    ]
    assert all(row.status == "unavailable" for row in eigenvector_rows)
    assert all("did not converge" in row.reason for row in eigenvector_rows)


def test_no_retained_edges_continue_with_defined_and_unavailable_metrics() -> None:
    _, _, results = _analysis(np.eye(3))
    assert _global(results, "number_of_edges", "binary").value == 0
    assert _global(results, "global_efficiency", "binary").value == 0.0
    assert _global(results, "mean_clustering_coefficient", "weighted").value == 0.0
    assert _global(results, "total_retained_positive_strength", "weighted").value == 0.0
    assert _global(results, "characteristic_path_length", "binary").status == "unavailable"
    assert _nodal(results, "R0", "degree", "binary").value == 0
    assert _nodal(results, "R0", "strength", "weighted").value == 0.0
    assert _nodal(results, "R0", "eigenvector_centrality", "weighted").status == "unavailable"


def _manual_graphs(nodes: tuple[str, ...]) -> ConstructedGraphs:
    binary = nx.Graph()
    weighted = nx.Graph()
    binary.add_nodes_from(nodes)
    weighted.add_nodes_from(nodes)
    diagnostics = GraphDiagnostics(
        number_of_nodes=len(nodes),
        number_of_possible_edges=len(nodes) * (len(nodes) - 1) // 2,
        positive_eligible_edges=0,
        negative_excluded_edges=0,
        zero_entries=0,
        retained_edges=0,
        target_edge_count=None,
        requested_density=None,
        achieved_density=0.0,
        threshold_mode="unthresholded",
        minimum_retained_weight=None,
        median_retained_weight=None,
        maximum_retained_weight=None,
        isolated_nodes=len(nodes),
        connected_components=len(nodes),
        largest_connected_component_size=1 if nodes else 0,
        connected=len(nodes) == 1,
        edge_weight_tie_count=0,
        boundary_tie_count=0,
        warnings=(),
    )
    return ConstructedGraphs(binary, weighted, (), diagnostics)


@pytest.mark.parametrize("nodes", [(), ("only",)])
def test_empty_and_single_node_global_edge_cases(nodes: tuple[str, ...]) -> None:
    metrics = compute_global_metrics(_manual_graphs(nodes), TopologyConfig(enabled=True), nodes)
    efficiency = next(
        row
        for row in metrics
        if row.metric == "global_efficiency" and row.representation == "binary"
    )
    path = next(
        row
        for row in metrics
        if row.metric == "characteristic_path_length" and row.representation == "binary"
    )
    assert efficiency.status == "unavailable"
    assert path.status == "unavailable"
    assert efficiency.value is None
    assert path.value is None


def test_output_tables_strict_json_and_collision(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _, _, results = _analysis(matrix)
    paths = write_topology_outputs(results, tmp_path)
    global_table = pd.read_csv(paths["global_metrics"], sep="\t")
    nodal_table = pd.read_csv(paths["nodal_metrics"], sep="\t")
    payload_text = paths["topology_results"].read_text(encoding="utf-8")
    payload = json.loads(payload_text, parse_constant=lambda value: pytest.fail(value))
    assert not global_table.empty
    assert not nodal_table.empty
    assert {"metric", "representation", "value", "status", "definition"}.issubset(
        global_table.columns
    )
    assert {"roi_index", "roi_id", "label", "metric", "status"}.issubset(nodal_table.columns)
    assert nodal_table.iloc[0]["label"] == "Region 0"
    assert payload["metadata"]["roi_order"] == ["R0", "R1", "R2"]
    assert "NaN" not in payload_text
    assert "Infinity" not in payload_text
    with pytest.raises(TopologyOutputError, match="Refusing"):
        write_topology_outputs(results, tmp_path)
    write_topology_outputs(results, tmp_path, overwrite=True)


def test_disabled_topology_is_meaningful_failure() -> None:
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    connectome, graphs, _ = _analysis(matrix)
    with pytest.raises(TopologyAnalysisError, match="disabled"):
        analyze_topology(graphs, connectome, TopologyConfig(), GraphConfig(enabled=True))

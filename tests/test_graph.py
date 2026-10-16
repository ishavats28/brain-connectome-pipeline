from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.config import GraphConfig
from brain_connectome_pipeline.graph import (
    GraphConstructionError,
    GraphOutputError,
    construct_graphs,
    write_graph_outputs,
)
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.models import ConnectomeData, MatrixSemantics


def _connectome(
    matrix: np.ndarray,
    semantics: MatrixSemantics = "correlation",
    *,
    labels: tuple[str | None, ...] | None = None,
    coordinates: tuple[tuple[float, float, float] | None, ...] | None = None,
) -> ConnectomeData:
    count = matrix.shape[0]
    roi_ids = tuple(f"R{index}" for index in range(count))
    return ConnectomeData(
        matrix=matrix,
        roi_ids=roi_ids,
        roi_labels=labels or (None,) * count,
        coordinates=coordinates or (None,) * count,
        atlas="Supplied atlas",
        input_type="matrix",
        matrix_semantics=semantics,
        source_path=Path("synthetic.npy"),
        label_source="test",
        diagnostics=validate_matrix(matrix, semantics, 1e-8, 1e-8),
    )


def _graph_config(
    mode: str = "unthresholded",
    density: float | None = None,
    *,
    allow_binary_subsampling: bool = False,
) -> GraphConfig:
    return GraphConfig(
        enabled=True,
        mode=mode,  # type: ignore[arg-type]
        target_density=density,
        allow_binary_subsampling=allow_binary_subsampling,
    )


def test_positive_graph_semantics_attributes_and_immutability() -> None:
    matrix = np.array(
        [
            [1.0, 0.9, -0.4, 0.0],
            [0.9, 1.0, 0.8, 0.7],
            [-0.4, 0.8, 1.0, 0.6],
            [0.0, 0.7, 0.6, 1.0],
        ]
    )
    original = matrix.copy()
    connectome = _connectome(
        matrix,
        labels=("One", None, "Three", "Four"),
        coordinates=((1.0, 2.0, 3.0), None, None, None),
    )
    result = construct_graphs(connectome, _graph_config())
    np.testing.assert_array_equal(matrix, original)
    assert set(result.binary_graph.edges) == {
        ("R0", "R1"),
        ("R1", "R2"),
        ("R1", "R3"),
        ("R2", "R3"),
    }
    assert not result.binary_graph.has_edge("R0", "R2")
    assert result.binary_graph["R0"]["R1"] == {}
    assert result.weighted_graph["R0"]["R1"]["weight"] == 0.9
    assert result.weighted_graph["R0"]["R1"]["distance"] == pytest.approx(1 / 0.9)
    assert result.binary_graph.nodes["R0"]["label"] == "One"
    assert result.binary_graph.nodes["R0"]["coordinates"] == (1.0, 2.0, 3.0)
    assert result.binary_graph.nodes["R0"]["atlas"] == "Supplied atlas"
    assert "label" not in result.binary_graph.nodes["R1"]
    assert result.diagnostics.negative_excluded_edges == 1
    assert result.diagnostics.zero_entries == 1


def test_positive_weighted_adjacency_is_not_normalized() -> None:
    matrix = np.array([[0.0, 4.5, 0.0], [4.5, 0.0, 2.0], [0.0, 2.0, 0.0]])
    result = construct_graphs(_connectome(matrix, "positive_weighted_adjacency"), _graph_config())
    assert result.weighted_graph["R0"]["R1"]["weight"] == 4.5
    assert result.diagnostics.minimum_retained_weight == 2.0
    assert result.diagnostics.median_retained_weight == 3.25
    assert result.diagnostics.maximum_retained_weight == 4.5
    assert result.diagnostics.connected


def test_binary_semantics_and_isolated_node_preservation() -> None:
    matrix = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    result = construct_graphs(_connectome(matrix, "binary_adjacency"), _graph_config())
    assert result.binary_graph.number_of_nodes() == 4
    assert result.binary_graph.degree["R3"] == 0
    assert all(data == {} for _, _, data in result.binary_graph.edges(data=True))
    assert all(data["weight"] == 1.0 for _, _, data in result.weighted_graph.edges(data=True))
    assert result.diagnostics.isolated_nodes == 1
    assert result.diagnostics.connected_components == 2
    assert result.diagnostics.largest_connected_component_size == 3
    assert not result.diagnostics.connected


def test_proportional_density_rounding_and_exact_count() -> None:
    matrix = np.eye(5)
    values = iter([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    for source in range(5):
        for target in range(source + 1, 5):
            matrix[source, target] = matrix[target, source] = next(values)
    result = construct_graphs(_connectome(matrix), _graph_config("proportional", 0.15))
    assert result.diagnostics.number_of_possible_edges == 10
    assert result.diagnostics.target_edge_count == 2
    assert result.diagnostics.retained_edges == 2
    assert result.diagnostics.requested_density == 0.15
    assert result.diagnostics.achieved_density == 0.2


def test_deterministic_tie_breaking_and_reproducibility() -> None:
    matrix = np.full((4, 4), 0.5)
    np.fill_diagonal(matrix, 1.0)
    config = _graph_config("proportional", 2 / 6)
    first = construct_graphs(_connectome(matrix), config)
    second = construct_graphs(_connectome(matrix), config)
    expected = [(0, 1), (0, 2)]
    assert [(edge.source_index, edge.target_index) for edge in first.selected_edges] == expected
    assert first.selected_edges == second.selected_edges
    assert first.diagnostics.edge_weight_tie_count == 6
    assert first.diagnostics.boundary_tie_count == 6
    assert any("ROI index" in warning for warning in first.diagnostics.warnings)


def test_insufficient_edges_and_zero_density() -> None:
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.8
    insufficient = construct_graphs(_connectome(matrix), _graph_config("proportional", 0.5))
    assert insufficient.diagnostics.target_edge_count == 3
    assert insufficient.diagnostics.retained_edges == 1
    assert insufficient.diagnostics.achieved_density == pytest.approx(1 / 6)
    assert any("Insufficient" in warning for warning in insufficient.diagnostics.warnings)
    empty = construct_graphs(_connectome(matrix), _graph_config("proportional", 0.0))
    assert empty.diagnostics.retained_edges == 0
    assert empty.diagnostics.achieved_density == 0.0
    assert empty.diagnostics.minimum_retained_weight is None
    assert empty.diagnostics.connected_components == 4


def test_unthresholded_mode_retains_every_eligible_edge() -> None:
    matrix = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.1], [0.2, 0.1, 1.0]])
    result = construct_graphs(_connectome(matrix), _graph_config())
    assert result.diagnostics.threshold_mode == "unthresholded"
    assert result.diagnostics.requested_density is None
    assert result.diagnostics.retained_edges == 3
    assert nx.is_connected(result.binary_graph)


def test_binary_proportional_preserves_topology_without_opt_in() -> None:
    matrix = np.ones((4, 4)) - np.eye(4)
    connectome = _connectome(matrix, "binary_adjacency")
    preserved = construct_graphs(connectome, _graph_config("proportional", 1 / 6))
    assert preserved.diagnostics.retained_edges == 6
    assert preserved.diagnostics.threshold_mode == "binary_topology_preserved"
    sampled = construct_graphs(
        connectome,
        _graph_config("proportional", 1 / 6, allow_binary_subsampling=True),
    )
    assert sampled.diagnostics.retained_edges == 1
    assert sampled.selected_edges[0].source_index == 0
    assert sampled.selected_edges[0].target_index == 1
    assert any("physiologically" in warning for warning in sampled.diagnostics.warnings)


def test_disabled_or_invalid_direct_configuration_fails() -> None:
    connectome = _connectome(np.eye(2))
    with pytest.raises(GraphConstructionError, match="disabled"):
        construct_graphs(connectome, GraphConfig())
    with pytest.raises(GraphConstructionError, match="density"):
        construct_graphs(connectome, _graph_config("proportional", 1.5))


def test_output_files_and_collision_safeguard(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 0.75, -0.2], [0.75, 1.0, 0.0], [-0.2, 0.0, 1.0]])
    connectome = _connectome(matrix, labels=("First", "Second", None))
    graphs = construct_graphs(connectome, _graph_config())
    paths = write_graph_outputs(graphs, connectome, tmp_path)
    assert set(paths) == {
        "binary_adjacency",
        "weighted_adjacency",
        "edge_list",
        "node_metadata",
        "diagnostics",
    }
    binary = pd.read_csv(paths["binary_adjacency"], sep="\t", index_col=0)
    weighted = pd.read_csv(paths["weighted_adjacency"], sep="\t", index_col=0)
    edges = pd.read_csv(paths["edge_list"], sep="\t")
    nodes = pd.read_csv(paths["node_metadata"], sep="\t")
    assert binary.loc["R0", "R1"] == 1
    assert binary.loc["R0", "R2"] == 0
    assert weighted.loc["R0", "R1"] == 0.75
    assert list(edges[["source", "target"]].iloc[0]) == ["R0", "R1"]
    assert edges.loc[0, "weight"] == 0.75
    assert nodes.loc[0, "label"] == "First"
    diagnostics = json.loads(paths["diagnostics"].read_text(encoding="utf-8"))
    assert diagnostics["retained_edges"] == 1
    with pytest.raises(GraphOutputError, match="Refusing"):
        write_graph_outputs(graphs, connectome, tmp_path)
    write_graph_outputs(graphs, connectome, tmp_path, overwrite=True)


def test_diagnostics_are_json_serializable() -> None:
    matrix = np.array([[1.0, -0.5], [-0.5, 1.0]])
    result = construct_graphs(_connectome(matrix), _graph_config())
    restored = json.loads(json.dumps(result.summary_dict()))
    assert restored["negative_excluded_edges"] == 1
    assert restored["retained_edges"] == 0


def test_reciprocal_distance_overflow_is_rejected() -> None:
    tiny = np.nextafter(0.0, 1.0)
    matrix = np.array([[1.0, tiny], [tiny, 1.0]])
    with pytest.raises(GraphConstructionError, match="distance|overflowed"):
        construct_graphs(_connectome(matrix), _graph_config())

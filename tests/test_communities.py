from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.communities import (
    CommunityAnalysisError,
    CommunityAssignment,
    CommunityOutputError,
    CommunityResults,
    compute_community_node_metrics,
    detect_communities,
    validate_partition,
    write_community_outputs,
)
from brain_connectome_pipeline.config import CommunityConfig, GraphConfig
from brain_connectome_pipeline.graph import construct_graphs
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.models import ConnectomeData


def _analysis(matrix: np.ndarray):
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
    graphs = construct_graphs(connectome, GraphConfig(enabled=True))
    return connectome, graphs


def _specified_partition(connectome: ConnectomeData, partition: dict[str, int]):
    sizes = {
        community: sum(value == community for value in partition.values())
        for community in set(partition.values())
    }
    return CommunityResults(
        assignments=tuple(
            CommunityAssignment(
                index,
                roi_id,
                connectome.roi_labels[index],
                partition[roi_id],
                sizes[partition[roi_id]],
            )
            for index, roi_id in enumerate(connectome.roi_ids)
        ),
        partition=partition,
        modularity=None,
        modularity_status="unavailable",
        modularity_reason="test partition",
        metadata={
            "algorithm": "specified test partition",
            "number_of_communities": len(sizes),
            "community_sizes": list(sizes.values()),
        },
        warnings=(),
    )


def test_weighted_louvain_is_reproducible_complete_and_stably_labeled() -> None:
    matrix = np.eye(7)
    for group in ((0, 1, 2), (3, 4, 5)):
        for source in group:
            for target in group:
                if source < target:
                    matrix[source, target] = matrix[target, source] = 0.9
    matrix[2, 3] = matrix[3, 2] = 0.05
    connectome, graphs = _analysis(matrix)
    original_matrix = matrix.copy()
    original_edges = {
        (source, target): dict(attributes)
        for source, target, attributes in graphs.weighted_graph.edges(data=True)
    }
    config = CommunityConfig(enabled=True, seed=23)
    first = detect_communities(graphs, connectome, config)
    second = detect_communities(graphs, connectome, config)
    assert first.partition == second.partition
    assert tuple(first.partition) == connectome.roi_ids
    assert len(first.assignments) == 7
    assert first.partition["R0"] == 0
    assert first.partition["R3"] == 1
    assert first.partition["R6"] == 2
    np.testing.assert_array_equal(matrix, original_matrix)
    assert {
        (source, target): dict(attributes)
        for source, target, attributes in graphs.weighted_graph.edges(data=True)
    } == original_edges
    assert first.modularity_status == "available"
    assert first.modularity == pytest.approx(
        nx.community.modularity(
            graphs.weighted_graph,
            [
                {node for node, value in first.partition.items() if value == community}
                for community in range(3)
            ],
            weight="weight",
            resolution=1.0,
        )
    )


def test_zero_edge_and_disconnected_graphs_preserve_isolates() -> None:
    connectome, graphs = _analysis(np.eye(3))
    results = detect_communities(graphs, connectome, CommunityConfig(enabled=True))
    assert results.partition == {"R0": 0, "R1": 1, "R2": 2}
    assert results.modularity is None
    assert results.modularity_status == "unavailable"
    assert any("zero-edge" in warning for warning in results.warnings)
    assert any("disconnected" in warning for warning in results.warnings)


def test_partition_validation_rejects_missing_extra_and_invalid_identifiers() -> None:
    _, graphs = _analysis(np.eye(2))
    with pytest.raises(CommunityAnalysisError, match="cover every"):
        validate_partition(graphs.binary_graph, {"R0": 0})
    with pytest.raises(CommunityAnalysisError, match="nonnegative integers"):
        validate_partition(graphs.binary_graph, {"R0": 0, "R1": True})


def test_binary_and_weighted_participation_use_different_edge_quantities() -> None:
    matrix = np.eye(4)
    for source, target, weight in ((0, 1, 0.9), (0, 2, 0.2), (0, 3, 0.2)):
        matrix[source, target] = matrix[target, source] = weight
    connectome, graphs = _analysis(matrix)
    communities = _specified_partition(connectome, {"R0": 0, "R1": 0, "R2": 1, "R3": 1})
    rows = compute_community_node_metrics(graphs, connectome, communities)
    assert rows[0].binary_participation == pytest.approx(4 / 9)
    assert rows[0].weighted_participation == pytest.approx(1 - (0.9 / 1.3) ** 2 - (0.4 / 1.3) ** 2)
    assert rows[1].binary_participation == 0.0
    assert rows[2].binary_participation == 0.0
    assert rows[2].weighted_participation == 0.0


def test_isolate_participation_and_zero_variance_zscores_are_unavailable() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 0.5
    connectome, graphs = _analysis(matrix)
    communities = _specified_partition(connectome, {"R0": 0, "R1": 0, "R2": 1})
    rows = compute_community_node_metrics(graphs, connectome, communities)
    assert rows[2].binary_participation is None
    assert rows[2].weighted_participation_status == "unavailable"
    assert "zero" in (rows[2].weighted_participation_reason or "")
    assert rows[0].within_module_degree_zscore is None
    assert "standard deviation" in (rows[0].within_module_degree_zscore_reason or "")
    assert "at least two" in (rows[2].within_module_degree_zscore_reason or "")


def test_within_module_degree_uses_population_zscore() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 0.8
    matrix[0, 2] = matrix[2, 0] = 0.4
    connectome, graphs = _analysis(matrix)
    communities = _specified_partition(connectome, {"R0": 0, "R1": 0, "R2": 0})
    rows = compute_community_node_metrics(graphs, connectome, communities)
    assert [row.within_module_degree for row in rows] == [2, 1, 1]
    assert rows[0].within_module_degree_zscore == pytest.approx(np.sqrt(2))
    assert rows[1].within_module_degree_zscore == pytest.approx(-1 / np.sqrt(2))
    assert all(row.binary_participation == 0.0 for row in rows)


def test_community_outputs_are_strict_json_and_collision_safe(tmp_path: Path) -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 0.7
    connectome, graphs = _analysis(matrix)
    results = detect_communities(graphs, connectome, CommunityConfig(enabled=True))
    metrics = compute_community_node_metrics(graphs, connectome, results)
    paths = write_community_outputs(results, tmp_path, node_metrics=metrics)

    def reject_constant(value: str):
        raise ValueError(value)

    json.loads(
        paths["community_summary"].read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    assignments = pd.read_csv(paths["community_assignments"], sep="\t")
    assert assignments["roi_id"].tolist() == list(connectome.roi_ids)
    with pytest.raises(CommunityOutputError, match="Refusing to overwrite"):
        write_community_outputs(results, tmp_path, node_metrics=metrics)

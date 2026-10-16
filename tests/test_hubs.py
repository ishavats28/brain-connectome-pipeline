from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain_connectome_pipeline.communities import (
    CommunityAssignment,
    CommunityResults,
    compute_community_node_metrics,
)
from brain_connectome_pipeline.config import CommunityConfig, GraphConfig, TopologyConfig
from brain_connectome_pipeline.graph import construct_graphs
from brain_connectome_pipeline.hubs import HubAnalysisError, characterize_hubs
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.metrics import analyze_topology
from brain_connectome_pipeline.models import ConnectomeData


def _inputs(matrix: np.ndarray):
    count = matrix.shape[0]
    connectome = ConnectomeData(
        matrix=matrix,
        roi_ids=tuple(f"R{index}" for index in range(count)),
        roi_labels=(None,) * count,
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
    topology = analyze_topology(graphs, connectome, TopologyConfig(enabled=True), graph_config)
    partition = {node: 0 for node in connectome.roi_ids}
    communities = CommunityResults(
        tuple(
            CommunityAssignment(index, node, None, 0, count)
            for index, node in enumerate(connectome.roi_ids)
        ),
        partition,
        None,
        "unavailable",
        "specified test partition",
        {
            "algorithm": "specified test partition",
            "number_of_communities": 1,
            "community_sizes": [count],
        },
        (),
    )
    community_metrics = compute_community_node_metrics(graphs, connectome, communities)
    return connectome, topology, community_metrics


def test_star_center_is_the_only_candidate_and_ranking_is_deterministic() -> None:
    matrix = np.eye(5)
    for leaf, weight in enumerate((0.9, 0.8, 0.7, 0.6), start=1):
        matrix[0, leaf] = matrix[leaf, 0] = weight
    connectome, topology, community_metrics = _inputs(matrix)
    config = CommunityConfig(
        enabled=True,
        hub_enabled=True,
        hub_percentile_threshold=0.85,
        hub_minimum_metrics=4,
    )
    first = characterize_hubs(topology, community_metrics, connectome, config)
    second = characterize_hubs(topology, community_metrics, connectome, config)
    assert first == second
    assert first.status == "available"
    assert [row.roi_id for row in first.rows if row.is_candidate_hub] == ["R0"]
    center = first.rows[0]
    assert center.combined_rank == 1.0
    assert center.degree_percentile_rank == 1.0
    assert center.betweenness_percentile_rank == 1.0
    assert all(row.participation_percentile_rank == 0.6 for row in first.rows)
    assert first.summary_dict()["candidate_count"] == 1


def test_equal_uninformative_criteria_do_not_manufacture_candidates() -> None:
    connectome, topology, community_metrics = _inputs(np.eye(4))
    results = characterize_hubs(
        topology,
        community_metrics,
        connectome,
        CommunityConfig(enabled=True, hub_enabled=True, hub_minimum_metrics=3),
    )
    assert results.status == "unavailable"
    assert "no between-node variation" in (results.reason or "")
    assert not any(row.is_candidate_hub for row in results.rows)
    assert all(row.weighted_participation is None for row in results.rows)


def test_unavailable_criteria_are_not_treated_as_zero() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 0.8
    connectome, topology, community_metrics = _inputs(matrix)
    results = characterize_hubs(
        topology,
        community_metrics,
        connectome,
        CommunityConfig(enabled=True, hub_enabled=True, hub_minimum_metrics=4),
    )
    isolate = results.rows[2]
    assert isolate.weighted_participation is None
    assert isolate.participation_percentile_rank is None
    assert isolate.eligibility_status == "ineligible"
    assert isolate.combined_score is None


def test_hub_analysis_rejects_disabled_or_misaligned_inputs() -> None:
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 0.8
    connectome, topology, community_metrics = _inputs(matrix)
    with pytest.raises(HubAnalysisError, match="disabled"):
        characterize_hubs(
            topology,
            community_metrics,
            connectome,
            CommunityConfig(enabled=True, hub_enabled=False),
        )
    with pytest.raises(HubAnalysisError, match="original ROI order"):
        characterize_hubs(
            topology,
            tuple(reversed(community_metrics)),
            connectome,
            CommunityConfig(enabled=True, hub_enabled=True),
        )

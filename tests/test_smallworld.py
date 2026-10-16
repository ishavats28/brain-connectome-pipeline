from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.config import GraphConfig, SmallWorldConfig
from brain_connectome_pipeline.graph import construct_graphs
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.models import ConnectomeData
from brain_connectome_pipeline.smallworld import (
    SmallWorldAnalysisError,
    SmallWorldOutputError,
    analyze_smallworld,
    calculate_smallworld_ratios,
    rewire_connected_null,
    write_smallworld_outputs,
)


def _constructed(graph: nx.Graph):
    nodes = tuple(str(node) for node in graph.nodes)
    relabeled = nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes})
    matrix = nx.to_numpy_array(relabeled, nodelist=list(nodes), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    connectome = ConnectomeData(
        matrix=matrix,
        roi_ids=nodes,
        roi_labels=(None,) * len(nodes),
        coordinates=(None,) * len(nodes),
        atlas=None,
        input_type="matrix",
        matrix_semantics="correlation",
        source_path=Path("synthetic.npy"),
        label_source="test",
        diagnostics=validate_matrix(matrix, "correlation", 1e-8, 1e-8),
    )
    return construct_graphs(connectome, GraphConfig(enabled=True))


def _config(**changes: int | bool) -> SmallWorldConfig:
    values = {
        "enabled": True,
        "ensemble_size": 3,
        "minimum_valid_samples": 2,
        "swaps_per_edge": 1,
        "maximum_ensemble_attempts": 12,
        "seed": 41,
    }
    values.update(changes)
    return SmallWorldConfig(**values)


def test_connected_null_preserves_degrees_connectivity_and_changes_edges() -> None:
    source = nx.watts_strogatz_graph(12, 4, 0.25, seed=4)
    randomized, successful = rewire_connected_null(source, requested_swaps=24, seed=101)
    assert successful > 0
    assert nx.is_connected(randomized)
    assert dict(randomized.degree()) == dict(source.degree())
    assert set(map(frozenset, randomized.edges)) != set(map(frozenset, source.edges))


def test_smallworld_ensemble_is_reproducible_and_uses_distinct_seeds() -> None:
    graphs = _constructed(nx.watts_strogatz_graph(12, 4, 0.25, seed=4))
    first = analyze_smallworld(graphs, _config())
    second = analyze_smallworld(graphs, _config())
    assert first == second
    assert first.status == "available"
    accepted = [sample for sample in first.samples if sample.status == "accepted"]
    assert len(accepted) == 3
    assert len({sample.seed for sample in accepted}) == 3
    assert first.observed_path_length == pytest.approx(
        nx.average_shortest_path_length(graphs.binary_graph)
    )
    assert first.sigma == pytest.approx(first.gamma / first.lambda_value)


def test_disconnected_graph_is_unavailable_without_null_attempts() -> None:
    graph = nx.Graph([(0, 1), (2, 3)])
    results = analyze_smallworld(_constructed(graph), _config())
    assert results.status == "unavailable"
    assert "connected observed graph" in (results.reason or "")
    assert results.samples == ()
    assert results.sigma is None


@pytest.mark.parametrize("graph", [nx.complete_graph(5), nx.star_graph(5)])
def test_graphs_without_meaningful_rewiring_are_unavailable(graph: nx.Graph) -> None:
    results = analyze_smallworld(
        _constructed(graph),
        _config(ensemble_size=2, minimum_valid_samples=1, maximum_ensemble_attempts=3),
    )
    assert results.status == "unavailable"
    assert results.metadata["accepted_samples"] == 0
    assert all(sample.status == "rejected" for sample in results.samples)
    assert all(sample.clustering is None for sample in results.samples)


def test_too_few_valid_nulls_remains_unavailable(monkeypatch) -> None:
    graphs = _constructed(nx.watts_strogatz_graph(10, 4, 0.2, seed=7))

    def fail_rewiring(*args, **kwargs):
        raise SmallWorldAnalysisError("bounded synthetic failure")

    monkeypatch.setattr("brain_connectome_pipeline.smallworld.rewire_connected_null", fail_rewiring)
    results = analyze_smallworld(
        graphs,
        _config(ensemble_size=3, minimum_valid_samples=3, maximum_ensemble_attempts=3),
    )
    assert results.status == "unavailable"
    assert results.metadata["accepted_samples"] == 0
    assert "required" in (results.reason or "")


def test_smallworld_ratio_arithmetic_and_denominator_guards() -> None:
    gamma, lambda_value, sigma = calculate_smallworld_ratios(0.4, 2.0, 0.2, 1.6)
    assert gamma == 2.0
    assert lambda_value == 1.25
    assert sigma == 1.6
    with pytest.raises(SmallWorldAnalysisError, match="clustering"):
        calculate_smallworld_ratios(0.4, 2.0, 0.0, 1.6)
    with pytest.raises(SmallWorldAnalysisError, match="finite"):
        calculate_smallworld_ratios(float("nan"), 2.0, 0.2, 1.6)


def test_smallworld_outputs_are_strict_and_collision_safe(tmp_path: Path) -> None:
    results = analyze_smallworld(
        _constructed(nx.watts_strogatz_graph(12, 4, 0.25, seed=4)), _config()
    )
    paths = write_smallworld_outputs(results, tmp_path)

    def reject_constant(value: str):
        raise ValueError(value)

    json.loads(
        paths["smallworld_summary"].read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    nulls = pd.read_csv(paths["smallworld_nulls"], sep="\t")
    assert len(nulls) == len(results.samples)
    assert set(nulls["status"]) <= {"accepted", "rejected"}
    with pytest.raises(SmallWorldOutputError, match="Refusing to overwrite"):
        write_smallworld_outputs(results, tmp_path)


def test_disabled_smallworld_analysis_fails_clearly() -> None:
    with pytest.raises(SmallWorldAnalysisError, match="disabled"):
        analyze_smallworld(_constructed(nx.cycle_graph(6)), SmallWorldConfig())

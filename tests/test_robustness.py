from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.config import GraphConfig, RobustnessConfig
from brain_connectome_pipeline.graph import construct_graphs
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.models import ConnectomeData, MatrixDiagnostics
from brain_connectome_pipeline.robustness import (
    RobustnessAnalysisError,
    RobustnessOutputError,
    analyze_robustness,
    removal_checkpoints,
    write_robustness_outputs,
)


def _inputs(matrix: np.ndarray):
    count = matrix.shape[0]
    diagnostics = (
        validate_matrix(matrix, "correlation", 1e-8, 1e-8)
        if count >= 2
        else MatrixDiagnostics(1e-8, 0.0, True, 1.0, 1.0, "unit", 0, 0, 0)
    )
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
        diagnostics=diagnostics,
    )
    graphs = construct_graphs(connectome, GraphConfig(enabled=True))
    return connectome, graphs


def _star(count: int = 5) -> np.ndarray:
    matrix = np.eye(count)
    for leaf in range(1, count):
        matrix[0, leaf] = matrix[leaf, 0] = 0.9 - leaf * 0.05
    return matrix


def _aggregate(results, strategy: str, checkpoint: int, metric: str):
    return next(
        row
        for row in results.aggregate_rows
        if row.strategy == strategy and row.checkpoint_index == checkpoint and row.metric == metric
    )


def test_half_up_checkpoints_deduplicate_small_network_fractions() -> None:
    checkpoints = removal_checkpoints(5, (0.0, 0.05, 0.1, 0.2, 0.25, 1.0))
    assert [checkpoint.removed_count for checkpoint in checkpoints] == [0, 1, 5]
    assert checkpoints[0].requested_fractions == (0.0, 0.05)
    assert checkpoints[1].requested_fractions == (0.1, 0.2, 0.25)
    assert checkpoints[2].actual_removed_fraction == 1.0


def test_targeted_star_removes_fixed_initial_strength_hub_first() -> None:
    connectome, graphs = _inputs(_star())
    results = analyze_robustness(
        graphs,
        connectome,
        RobustnessConfig(
            enabled=True,
            ranking_criterion="strength",
            removal_fractions=(0.0, 0.2, 0.4),
            random_trials=4,
            seed=17,
        ),
    )
    assert results.ranking[0].roi_id == "R0"
    targeted = [row for row in results.trial_rows if row.strategy == "targeted"]
    assert targeted[1].removed_nodes == ("R0",)
    assert targeted[2].removed_nodes[:1] == targeted[1].removed_nodes
    assert targeted[1].largest_component_fraction == pytest.approx(0.2)
    assert targeted[1].connected_components == 4
    assert targeted[1].survivor_efficiency == 0.0
    assert targeted[1].original_normalized_efficiency == 0.0
    assert targeted[0].survivor_efficiency == pytest.approx(0.7)
    assert targeted[0].original_normalized_efficiency == pytest.approx(0.7)


def test_survivor_and_original_normalized_efficiency_are_distinct() -> None:
    connectome, graphs = _inputs(_star())
    results = analyze_robustness(
        graphs,
        connectome,
        RobustnessConfig(
            enabled=True,
            removal_fractions=(0.0, 0.2),
            random_trials=12,
            seed=3,
        ),
    )
    leaf_removal = next(
        row
        for row in results.trial_rows
        if row.strategy == "random" and row.removed_count == 1 and row.removed_nodes != ("R0",)
    )
    assert leaf_removal.survivor_efficiency == pytest.approx(0.75)
    assert leaf_removal.original_normalized_efficiency == pytest.approx(0.45)


def test_random_sequences_and_aggregates_are_reproducible_and_auditable() -> None:
    connectome, graphs = _inputs(_star(6))
    config = RobustnessConfig(
        enabled=True,
        ranking_criterion="degree",
        removal_fractions=(0.0, 0.2, 0.5),
        random_trials=5,
        seed=29,
    )
    first = analyze_robustness(graphs, connectome, config)
    second = analyze_robustness(graphs, connectome, config)
    assert first == second
    assert first.metadata["trial_seeds"] == [29, 30, 31, 32, 33]
    random_rows = [
        row for row in first.trial_rows if row.strategy == "random" and row.checkpoint_index == 1
    ]
    values = [row.largest_component_fraction for row in random_rows]
    aggregate = _aggregate(first, "random_mean", 1, "largest_component_fraction")
    assert aggregate.value == pytest.approx(statistics.fmean(values))
    assert aggregate.sample_standard_deviation == pytest.approx(statistics.stdev(values))
    for trial_id in {row.trial_id for row in first.trial_rows if row.strategy == "random"}:
        trial = [row for row in first.trial_rows if row.trial_id == trial_id]
        assert trial[0].removed_nodes == ()
        assert trial[1].removed_nodes == trial[2].removed_nodes[: trial[1].removed_count]


def test_disconnected_graph_is_not_repaired_and_full_removal_is_explicit() -> None:
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.8
    connectome, graphs = _inputs(matrix)
    results = analyze_robustness(
        graphs,
        connectome,
        RobustnessConfig(
            enabled=True,
            removal_fractions=(0.0, 1.0),
            random_trials=2,
        ),
    )
    baseline = next(
        row for row in results.trial_rows if row.strategy == "targeted" and row.removed_count == 0
    )
    removed_all = next(
        row for row in results.trial_rows if row.strategy == "targeted" and row.removed_count == 4
    )
    assert baseline.connected_components == 3
    assert baseline.largest_component_fraction == 0.5
    assert removed_all.retained_node_fraction == 0.0
    assert removed_all.largest_component_fraction == 0.0
    assert removed_all.survivor_efficiency is None
    assert removed_all.original_normalized_efficiency == 0.0
    assert any("disconnected" in warning for warning in results.warnings)


def test_single_node_and_disabled_analyses_are_unavailable_or_fail_clearly() -> None:
    connectome, graphs = _inputs(np.ones((1, 1)))
    unavailable = analyze_robustness(
        graphs, connectome, RobustnessConfig(enabled=True, removal_fractions=(0.0,))
    )
    assert unavailable.status == "unavailable"
    assert unavailable.trial_rows == ()
    with pytest.raises(RobustnessAnalysisError, match="disabled"):
        analyze_robustness(graphs, connectome, RobustnessConfig())


def test_robustness_outputs_are_strict_nonempty_and_collision_safe(tmp_path: Path) -> None:
    connectome, graphs = _inputs(_star())
    results = analyze_robustness(
        graphs,
        connectome,
        RobustnessConfig(
            enabled=True,
            removal_fractions=(0.0, 0.2, 0.4),
            random_trials=3,
            seed=5,
        ),
    )
    paths = write_robustness_outputs(results, tmp_path, plots_enabled=False)

    def reject_constant(value: str):
        raise ValueError(value)

    json.loads(
        paths["robustness_summary_json"].read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    trials = pd.read_csv(paths["robustness_trials"], sep="\t")
    summary = pd.read_csv(paths["robustness_summary"], sep="\t")
    ranking = pd.read_csv(paths["robustness_ranking"], sep="\t")
    assert len(trials) == 12
    assert not summary.empty
    assert ranking.iloc[0]["roi_id"] == "R0"
    with pytest.raises(RobustnessOutputError, match="Refusing to overwrite"):
        write_robustness_outputs(results, tmp_path, plots_enabled=False)


def test_robustness_plots_are_nonempty_pngs(tmp_path: Path) -> None:
    connectome, graphs = _inputs(_star())
    results = analyze_robustness(
        graphs,
        connectome,
        RobustnessConfig(
            enabled=True,
            removal_fractions=(0.0, 0.2, 0.4),
            random_trials=3,
            seed=5,
        ),
    )
    paths = write_robustness_outputs(results, tmp_path, plots_enabled=True)
    for key in ("robustness_lcc_plot", "robustness_efficiency_plot"):
        image = paths[key]
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 1_000

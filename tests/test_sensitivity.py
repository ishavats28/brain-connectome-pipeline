from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.config import DensitySensitivityConfig, GraphConfig
from brain_connectome_pipeline.io import validate_matrix
from brain_connectome_pipeline.models import ConnectomeData
from brain_connectome_pipeline.sensitivity import (
    SensitivityAnalysisError,
    SensitivityOutputError,
    analyze_density_sensitivity,
    write_sensitivity_outputs,
)


def _connectome(matrix: np.ndarray) -> ConnectomeData:
    count = matrix.shape[0]
    return ConnectomeData(
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


def _complete_weighted_matrix(count: int) -> np.ndarray:
    matrix = np.eye(count)
    weight = 0.95
    for source in range(count):
        for target in range(source + 1, count):
            matrix[source, target] = matrix[target, source] = weight
            weight -= 0.05
    return matrix


def _metric(results, density: float, metric: str, representation: str):
    return next(
        row
        for row in results.rows
        if row.requested_density == density
        and row.metric == metric
        and row.representation == representation
    )


def test_independent_density_graphs_have_expected_monotonic_edge_counts() -> None:
    matrix = _complete_weighted_matrix(5)
    original = matrix.copy()
    results = analyze_density_sensitivity(
        _connectome(matrix),
        GraphConfig(enabled=True),
        DensitySensitivityConfig(enabled=True, densities=(0.1, 0.2, 0.3)),
    )
    assert [record["retained_edges"] for record in results.density_records] == [1, 2, 3]
    assert [record["achieved_density"] for record in results.density_records] == [
        0.1,
        0.2,
        0.3,
    ]
    np.testing.assert_array_equal(matrix, original)
    assert len(results.rows) == 24
    assert [row.density_index for row in results.rows[:8]] == [0] * 8


def test_known_sparse_efficiency_and_undefined_path_length() -> None:
    matrix = _complete_weighted_matrix(5)
    results = analyze_density_sensitivity(
        _connectome(matrix),
        GraphConfig(enabled=True),
        DensitySensitivityConfig(enabled=True, densities=(0.1,)),
    )
    assert _metric(results, 0.1, "global_efficiency", "binary").value == pytest.approx(0.1)
    assert _metric(results, 0.1, "global_efficiency", "weighted").value == pytest.approx(0.095)
    path = _metric(results, 0.1, "characteristic_path_length", "binary")
    assert path.status == "unavailable"
    assert path.value is None
    assert "connected" in (path.reason or "")
    assert _metric(results, 0.1, "connected_components", "binary").value == 4
    assert _metric(results, 0.1, "largest_component_fraction", "binary").value == 0.4


def test_repeated_topologies_and_insufficient_edges_are_explicit() -> None:
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.8
    results = analyze_density_sensitivity(
        _connectome(matrix),
        GraphConfig(enabled=True),
        DensitySensitivityConfig(enabled=True, densities=(0.2, 0.5, 0.8)),
    )
    assert [record["retained_edges"] for record in results.density_records] == [1, 1, 1]
    assert len({record["topology_id"] for record in results.density_records}) == 1
    assert all(record["repeated_topology"] for record in results.density_records)
    summary = results.summary_dict()
    assert summary["unique_topologies"] == 1
    assert summary["repeated_topologies"][0]["requested_densities"] == [0.2, 0.5, 0.8]
    assert any("Insufficient" in warning for warning in results.warnings)


def test_density_sensitivity_is_reproducible_and_disabled_state_is_clear() -> None:
    connectome = _connectome(_complete_weighted_matrix(5))
    config = DensitySensitivityConfig(enabled=True, densities=(0.1, 0.2))
    first = analyze_density_sensitivity(connectome, GraphConfig(enabled=True), config)
    second = analyze_density_sensitivity(connectome, GraphConfig(enabled=True), config)
    assert first == second
    with pytest.raises(SensitivityAnalysisError, match="disabled"):
        analyze_density_sensitivity(
            connectome, GraphConfig(enabled=True), DensitySensitivityConfig()
        )


def test_sensitivity_outputs_are_strict_ordered_and_collision_safe(tmp_path: Path) -> None:
    results = analyze_density_sensitivity(
        _connectome(_complete_weighted_matrix(5)),
        GraphConfig(enabled=True),
        DensitySensitivityConfig(enabled=True, densities=(0.1, 0.2)),
    )
    paths = write_sensitivity_outputs(results, tmp_path, plots_enabled=False)

    def reject_constant(value: str):
        raise ValueError(value)

    summary = json.loads(
        paths["density_sensitivity_summary"].read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    table = pd.read_csv(paths["density_sensitivity"], sep="\t")
    assert summary["density_points"] == 2
    assert len(table) == 16
    assert table.iloc[0]["metric"] == "global_efficiency"
    assert table.iloc[0]["representation"] == "binary"
    with pytest.raises(SensitivityOutputError, match="Refusing to overwrite"):
        write_sensitivity_outputs(results, tmp_path, plots_enabled=False)


def test_sensitivity_plots_are_nonempty_pngs(tmp_path: Path) -> None:
    results = analyze_density_sensitivity(
        _connectome(_complete_weighted_matrix(5)),
        GraphConfig(enabled=True),
        DensitySensitivityConfig(enabled=True, densities=(0.1, 0.2, 0.3)),
    )
    paths = write_sensitivity_outputs(results, tmp_path, plots_enabled=True)
    for key in ("density_efficiency_plot", "density_clustering_plot"):
        image = paths[key]
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert image.stat().st_size > 1_000

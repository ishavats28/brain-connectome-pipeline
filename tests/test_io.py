from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from brain_connectome_pipeline.config import load_config
from brain_connectome_pipeline.io import InputValidationError, load_matrix_connectome


def _save(path: Path, matrix: np.ndarray) -> None:
    if path.suffix == ".npy":
        np.save(path, matrix)
    else:
        np.savetxt(path, matrix, delimiter="," if path.suffix == ".csv" else "\t")


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".npy"])
def test_load_supported_matrix_formats(
    tmp_path: Path, config_writer, signed_correlation: np.ndarray, suffix: str
) -> None:
    path = tmp_path / f"matrix{suffix}"
    _save(path, signed_correlation)
    result = load_matrix_connectome(load_config(config_writer(path)))
    np.testing.assert_array_equal(result.matrix, signed_correlation)
    assert result.diagnostics.negative_edges == 1
    assert result.diagnostics.positive_edges == 1
    assert result.diagnostics.zero_edges == 1
    assert result.label_source == "generated placeholders"
    assert result.roi_ids == ("ROI_0001", "ROI_0002", "ROI_0003")
    assert "not anatomical" in result.warnings[0]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 3)), "square"),
        (np.array([[1.0, 0.2], [0.4, 1.0]]), "asymmetric"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "nonfinite"),
        (np.array([[1.0, 1.2], [1.2, 1.0]]), "within"),
        (np.array([[0.0, 0.2], [0.2, 1.0]]), "diagonal"),
    ],
)
def test_reject_invalid_correlation_matrices(
    tmp_path: Path, config_writer, matrix: np.ndarray, message: str
) -> None:
    path = tmp_path / "matrix.npy"
    np.save(path, matrix)
    with pytest.raises(InputValidationError, match=message):
        load_matrix_connectome(load_config(config_writer(path)))


def test_harmless_rounding_asymmetry_is_preserved(tmp_path: Path, config_writer) -> None:
    matrix = np.array([[1.0, 0.2], [0.2 + 1e-10, 1.0]])
    path = tmp_path / "rounding.npy"
    np.save(path, matrix)
    result = load_matrix_connectome(load_config(config_writer(path)))
    assert result.matrix[1, 0] == 0.2 + 1e-10
    assert result.diagnostics.maximum_asymmetry == pytest.approx(1e-10)


def test_positive_weighted_adjacency_allows_weights_above_one(
    tmp_path: Path, config_writer
) -> None:
    matrix = np.array([[4.0, 2.5], [2.5, 3.0]])
    path = tmp_path / "weighted.npy"
    np.save(path, matrix)
    result = load_matrix_connectome(
        load_config(config_writer(path, semantics="positive_weighted_adjacency"))
    )
    assert result.matrix[0, 1] == 2.5
    assert result.diagnostics.diagonal_status == "recorded_nonzero"


def test_positive_weighted_adjacency_rejects_negative_edges(tmp_path: Path, config_writer) -> None:
    path = tmp_path / "weighted.npy"
    np.save(path, np.array([[0.0, -0.1], [-0.1, 0.0]]))
    with pytest.raises(InputValidationError, match="negative"):
        load_matrix_connectome(
            load_config(config_writer(path, semantics="positive_weighted_adjacency"))
        )


@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_binary_adjacency_accepts_binary_edges(tmp_path: Path, config_writer, edge: float) -> None:
    path = tmp_path / "binary.npy"
    np.save(path, np.array([[0.0, edge], [edge, 0.0]]))
    result = load_matrix_connectome(load_config(config_writer(path, semantics="binary_adjacency")))
    assert result.matrix[0, 1] == edge


def test_binary_adjacency_does_not_binarize(tmp_path: Path, config_writer) -> None:
    path = tmp_path / "binary.npy"
    np.save(path, np.array([[0.0, 0.7], [0.7, 0.0]]))
    with pytest.raises(InputValidationError, match="0 or 1"):
        load_matrix_connectome(load_config(config_writer(path, semantics="binary_adjacency")))


def test_embedded_labels_require_exact_row_column_order(tmp_path: Path, config_writer) -> None:
    path = tmp_path / "labeled.csv"
    path.write_text("roi,A,B\nB,1,0.2\nA,0.2,1\n", encoding="utf-8")
    config = load_config(config_writer(path, extra_input="  embedded_labels: true\n"))
    with pytest.raises(InputValidationError, match="exact order"):
        load_matrix_connectome(config)


@pytest.mark.parametrize(
    "header",
    ["roi,A,A\nA,1,0\nA,0,1\n", "roi,A,\nA,1,0\nB,0,1\n"],
)
def test_embedded_labels_reject_duplicates_and_missing(
    tmp_path: Path, config_writer, header: str
) -> None:
    path = tmp_path / "labeled.csv"
    path.write_text(header, encoding="utf-8")
    config = load_config(config_writer(path, extra_input="  embedded_labels: true\n"))
    with pytest.raises(InputValidationError, match="unique|missing"):
        load_matrix_connectome(config)


def test_external_roi_metadata_and_atlas(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0.4\n0.4,1\n", encoding="utf-8")
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "roi_id\tlabel\tx\ty\tz\nr2\tRight\t1\t2\t3\nr1\tLeft\t\t\t\n", encoding="utf-8"
    )
    config = load_config(
        config_writer(
            matrix,
            extra_input=f"  labels_path: {labels.as_posix()}\n  atlas: Supplied atlas\n",
        )
    )
    result = load_matrix_connectome(config)
    assert result.roi_ids == ("r2", "r1")
    assert result.roi_labels == ("Right", "Left")
    assert result.coordinates == ((1.0, 2.0, 3.0), None)
    assert result.atlas == "Supplied atlas"


@pytest.mark.parametrize(
    "rows",
    [
        "roi_id\nA\n",
        "roi_id\nA\nA\n",
        "roi_id\nA\n\n",
    ],
)
def test_missing_duplicate_or_mismatched_metadata(tmp_path: Path, config_writer, rows: str) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text(rows, encoding="utf-8")
    config = load_config(config_writer(matrix, extra_input=f"  labels_path: {labels.as_posix()}\n"))
    with pytest.raises(InputValidationError):
        load_matrix_connectome(config)


def test_inspection_diagnostics_are_json_serializable(
    tmp_path: Path, config_writer, signed_correlation: np.ndarray
) -> None:
    path = tmp_path / "matrix.npy"
    np.save(path, signed_correlation)
    data = load_matrix_connectome(load_config(config_writer(path))).inspection_dict()
    restored = json.loads(json.dumps(data))
    assert restored["diagnostics"]["negative_edges"] == 1


@pytest.mark.parametrize("name", ["matrix.txt", "matrix.npy"])
def test_unsupported_or_corrupted_matrix_has_helpful_error(
    tmp_path: Path, config_writer, name: str
) -> None:
    path = tmp_path / name
    path.write_text("not a matrix", encoding="utf-8")
    config = load_config(config_writer(path))
    with pytest.raises(InputValidationError, match="Unsupported|Could not read"):
        load_matrix_connectome(config)

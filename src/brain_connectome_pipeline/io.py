"""Connectivity-matrix and ROI metadata ingestion."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from brain_connectome_pipeline.config import InputConfig, PipelineConfig
from brain_connectome_pipeline.models import ConnectomeData, MatrixDiagnostics, MatrixSemantics


class InputValidationError(ValueError):
    """Raised when a scientific input fails explicit validation."""


def _delimiter(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    if path.suffix.lower() == ".tsv":
        return "\t"
    raise InputValidationError(f"Unsupported tabular format '{path.suffix}'; use CSV or TSV.")


def _read_matrix(config: InputConfig) -> tuple[NDArray[np.float64], tuple[str, ...] | None]:
    path = config.path
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            if config.embedded_labels:
                raise InputValidationError("NPY matrices cannot contain embedded ROI labels.")
            raw = np.load(path, allow_pickle=False)
            matrix = np.asarray(raw, dtype=np.float64)
            return matrix, None
        if suffix not in {".csv", ".tsv"}:
            raise InputValidationError(
                f"Unsupported matrix format '{suffix}'; use .csv, .tsv, or .npy."
            )
        embedded_columns: tuple[str, ...] | None = None
        if config.embedded_labels:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle, delimiter=_delimiter(path)))
            embedded_columns = tuple(value.strip() for value in header[1:])
            if not embedded_columns or not all(embedded_columns):
                raise InputValidationError("Embedded ROI identifiers cannot be missing or blank.")
            if len(set(embedded_columns)) != len(embedded_columns):
                raise InputValidationError("Embedded ROI identifiers must be unique.")
        frame = pd.read_csv(
            path,
            sep=_delimiter(path),
            index_col=0 if config.embedded_labels else None,
            header=0 if config.embedded_labels else None,
        )
        embedded: tuple[str, ...] | None = None
        if config.embedded_labels:
            if frame.index.isna().any():
                raise InputValidationError("Embedded ROI identifiers cannot be missing or blank.")
            row_ids = tuple(str(value).strip() for value in frame.index)
            assert embedded_columns is not None
            column_ids = embedded_columns
            if not all(row_ids) or not all(column_ids):
                raise InputValidationError("Embedded ROI identifiers cannot be missing or blank.")
            if len(set(row_ids)) != len(row_ids) or len(set(column_ids)) != len(column_ids):
                raise InputValidationError("Embedded ROI identifiers must be unique.")
            if row_ids != column_ids:
                raise InputValidationError(
                    "Embedded matrix row and column ROI identifiers must match in exact order."
                )
            embedded = row_ids
        matrix = frame.to_numpy(dtype=np.float64)
        return matrix, embedded
    except InputValidationError:
        raise
    except (OSError, ValueError, StopIteration, csv.Error, pd.errors.ParserError) as exc:
        raise InputValidationError(f"Could not read matrix '{path}': {exc}") from exc


def _read_roi_metadata(
    path: Path, expected: int
) -> tuple[tuple[str, ...], tuple[str | None, ...], tuple[tuple[float, float, float] | None, ...]]:
    try:
        frame = pd.read_csv(
            path, sep=_delimiter(path), dtype={"roi_id": "string", "label": "string"}
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise InputValidationError(f"Could not read ROI metadata '{path}': {exc}") from exc
    allowed = {"roi_id", "label", "x", "y", "z"}
    unknown = sorted(set(frame.columns) - allowed)
    if unknown:
        raise InputValidationError(f"Unknown ROI metadata column(s): {', '.join(unknown)}")
    if "roi_id" not in frame:
        raise InputValidationError("ROI metadata must contain a 'roi_id' column.")
    if len(frame) != expected:
        raise InputValidationError(
            f"ROI metadata has {len(frame)} rows; matrix requires {expected}."
        )
    if frame["roi_id"].isna().any():
        raise InputValidationError("ROI identifiers cannot be missing.")
    ids = tuple(str(value).strip() for value in frame["roi_id"])
    if not all(ids):
        raise InputValidationError("ROI identifiers cannot be blank.")
    if len(set(ids)) != len(ids):
        raise InputValidationError("ROI identifiers must be unique.")
    labels = (
        tuple(None if pd.isna(value) else str(value) for value in frame["label"])
        if "label" in frame
        else (None,) * expected
    )
    coordinate_columns = {"x", "y", "z"}.intersection(frame.columns)
    if coordinate_columns and coordinate_columns != {"x", "y", "z"}:
        raise InputValidationError("ROI coordinates require all three columns: x, y, and z.")
    coordinates: tuple[tuple[float, float, float] | None, ...] = (None,) * expected
    if coordinate_columns:
        try:
            values = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        except ValueError as exc:
            raise InputValidationError("ROI coordinates must be numeric.") from exc
        complete = np.isfinite(values).all(axis=1)
        empty = frame[["x", "y", "z"]].isna().all(axis=1).to_numpy()
        if not np.all(complete | empty):
            raise InputValidationError(
                "Each ROI coordinate must be either complete and finite or absent."
            )
        coordinates = tuple(
            tuple(map(float, row)) if ok else None for row, ok in zip(values, complete, strict=True)
        )
    return ids, labels, coordinates


def validate_matrix(
    matrix: NDArray[np.float64],
    semantics: MatrixSemantics,
    symmetry_tolerance: float,
    correlation_tolerance: float,
) -> MatrixDiagnostics:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InputValidationError(
            f"Connectivity matrix must be square; received shape {matrix.shape}."
        )
    if matrix.shape[0] < 2:
        raise InputValidationError("Connectivity matrix must contain at least two ROIs.")
    if not np.isfinite(matrix).all():
        raise InputValidationError("Connectivity matrix contains nonfinite or missing values.")
    maximum_asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if maximum_asymmetry > symmetry_tolerance:
        raise InputValidationError(
            f"Matrix is asymmetric (maximum |A-A.T|={maximum_asymmetry:.6g}); "
            f"tolerance is {symmetry_tolerance:.6g}."
        )
    diagonal = np.diag(matrix)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if semantics == "correlation":
        minimum = float(matrix.min())
        maximum = float(matrix.max())
        if minimum < -1.0 - correlation_tolerance or maximum > 1.0 + correlation_tolerance:
            raise InputValidationError(
                "Correlation values must lie within [-1, 1] "
                f"(observed {minimum:.6g} to {maximum:.6g})."
            )
        deviation = float(np.max(np.abs(diagonal - 1.0)))
        if deviation > correlation_tolerance:
            raise InputValidationError(
                f"Correlation diagonal must be unit valued; maximum deviation is {deviation:.6g}."
            )
        diagonal_status = "unit"
    elif semantics == "positive_weighted_adjacency":
        if np.any(upper < 0):
            raise InputValidationError(
                "Positive weighted adjacency has negative off-diagonal weights."
            )
        diagonal_status = (
            "zero" if np.all(np.abs(diagonal) <= correlation_tolerance) else "recorded_nonzero"
        )
    elif semantics == "binary_adjacency":
        valid = np.isclose(upper, 0.0, atol=correlation_tolerance, rtol=0.0) | np.isclose(
            upper, 1.0, atol=correlation_tolerance, rtol=0.0
        )
        if not np.all(valid):
            raise InputValidationError("Binary adjacency off-diagonal entries must be 0 or 1.")
        diagonal_status = (
            "zero" if np.all(np.abs(diagonal) <= correlation_tolerance) else "recorded_nonzero"
        )
    else:  # pragma: no cover - protected by typed configuration
        raise InputValidationError(f"Unsupported matrix semantics: {semantics}")
    return MatrixDiagnostics(
        symmetry_tolerance=symmetry_tolerance,
        maximum_asymmetry=maximum_asymmetry,
        symmetric=True,
        diagonal_minimum=float(diagonal.min()),
        diagonal_maximum=float(diagonal.max()),
        diagonal_status=diagonal_status,
        positive_edges=int(np.count_nonzero(upper > 0)),
        negative_edges=int(np.count_nonzero(upper < 0)),
        zero_edges=int(np.count_nonzero(upper == 0)),
    )


def load_matrix_connectome(config: PipelineConfig) -> ConnectomeData:
    matrix, embedded_ids = _read_matrix(config.input)
    diagnostics = validate_matrix(
        matrix,
        config.input.matrix_semantics,
        config.input.symmetry_tolerance,
        config.input.correlation_tolerance,
    )
    n_rois = matrix.shape[0]
    warnings: list[str] = []
    if config.input.labels_path is not None:
        roi_ids, roi_labels, coordinates = _read_roi_metadata(config.input.labels_path, n_rois)
        label_source = str(config.input.labels_path)
    elif embedded_ids is not None:
        roi_ids = embedded_ids
        roi_labels = (None,) * n_rois
        coordinates = (None,) * n_rois
        label_source = "embedded row/column identifiers"
    else:
        roi_ids = tuple(f"ROI_{index:04d}" for index in range(1, n_rois + 1))
        roi_labels = (None,) * n_rois
        coordinates = (None,) * n_rois
        label_source = "generated placeholders"
        warnings.append("ROI identifiers are generated placeholders, not anatomical region labels.")
    return ConnectomeData(
        matrix=matrix,
        roi_ids=roi_ids,
        roi_labels=roi_labels,
        coordinates=coordinates,
        atlas=config.input.atlas,
        input_type="matrix",
        matrix_semantics=config.input.matrix_semantics,
        source_path=config.input.path,
        label_source=label_source,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
    )

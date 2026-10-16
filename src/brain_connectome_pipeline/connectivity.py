"""Ordinary Pearson connectivity from preprocessed ROI time series."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from brain_connectome_pipeline.config import PipelineConfig
from brain_connectome_pipeline.io import (
    InputValidationError,
    _delimiter,
    _read_roi_metadata,
    validate_matrix,
)
from brain_connectome_pipeline.models import ConnectomeData


def _header(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.reader(handle, delimiter=_delimiter(path)))
    except (OSError, StopIteration, csv.Error) as exc:
        raise InputValidationError(
            f"Could not read time-series header from '{path}': {exc}"
        ) from exc
    values = tuple(value.strip() for value in row)
    if not values or not all(values):
        raise InputValidationError("Time-series ROI identifiers cannot be missing or blank.")
    if len(set(values)) != len(values):
        raise InputValidationError("Time-series ROI identifiers must be unique.")
    return values


def load_timeseries_connectome(config: PipelineConfig) -> ConnectomeData:
    path = config.input.path
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise InputValidationError("Time-series input must be CSV or TSV.")
    roi_ids = _header(path)
    try:
        frame = pd.read_csv(path, sep=_delimiter(path))
        values = frame.to_numpy(dtype=np.float64)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise InputValidationError(f"Could not read numeric time series '{path}': {exc}") from exc
    if values.ndim != 2 or values.shape[1] < 2:
        raise InputValidationError("Time-series table must contain at least two ROI columns.")
    if values.shape[0] < config.connectivity.minimum_observations:
        raise InputValidationError(
            f"Time-series table has {values.shape[0]} observations; at least "
            f"{config.connectivity.minimum_observations} are required."
        )
    if not np.isfinite(values).all():
        raise InputValidationError("Time-series table contains nonfinite or missing values.")
    ranges = np.ptp(values, axis=0)
    scales = np.maximum(1.0, np.max(np.abs(values), axis=0))
    bad = ranges <= config.connectivity.constant_tolerance * scales
    if np.any(bad):
        names = ", ".join(roi_ids[index] for index in np.flatnonzero(bad))
        raise InputValidationError(f"Constant or near-constant ROI time series: {names}.")
    matrix = np.asarray(np.corrcoef(values, rowvar=False), dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise InputValidationError("Pearson correlation produced undefined values.")
    diagnostics = validate_matrix(
        matrix,
        "correlation",
        config.input.symmetry_tolerance,
        config.input.correlation_tolerance,
    )
    labels: tuple[str | None, ...] = (None,) * len(roi_ids)
    coordinates: tuple[tuple[float, float, float] | None, ...] = (None,) * len(roi_ids)
    label_source = "time-series column identifiers"
    if config.input.labels_path is not None:
        metadata_ids, labels, coordinates = _read_roi_metadata(
            config.input.labels_path, len(roi_ids)
        )
        if metadata_ids != roi_ids:
            raise InputValidationError(
                "ROI metadata identifiers must exactly match time-series columns in order."
            )
        label_source = f"time-series columns aligned with {config.input.labels_path}"
    return ConnectomeData(
        matrix=matrix,
        roi_ids=roi_ids,
        roi_labels=labels,
        coordinates=coordinates,
        atlas=config.input.atlas,
        input_type="timeseries",
        matrix_semantics="correlation",
        source_path=path,
        label_source=label_source,
        diagnostics=diagnostics,
        number_of_observations=int(values.shape[0]),
        estimator_metadata={
            "estimator": "ordinary sample Pearson correlation",
            "orientation": "rows=observations, columns=ROIs",
            "fisher_z_transformed": False,
            "preprocessing": (
                "input assumed preprocessed upstream; no confound control performed here"
            ),
        },
    )


def load_connectome(config: PipelineConfig) -> ConnectomeData:
    if config.input.source_type == "matrix":
        from brain_connectome_pipeline.io import load_matrix_connectome

        return load_matrix_connectome(config)
    return load_timeseries_connectome(config)

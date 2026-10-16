"""Typed structures for original, unmodified connectome inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

MatrixSemantics = Literal["correlation", "positive_weighted_adjacency", "binary_adjacency"]
InputType = Literal["matrix", "timeseries"]


@dataclass(frozen=True)
class MatrixDiagnostics:
    symmetry_tolerance: float
    maximum_asymmetry: float
    symmetric: bool
    diagonal_minimum: float
    diagonal_maximum: float
    diagonal_status: str
    positive_edges: int
    negative_edges: int
    zero_edges: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symmetry_tolerance": self.symmetry_tolerance,
            "maximum_asymmetry": self.maximum_asymmetry,
            "symmetric": self.symmetric,
            "diagonal_minimum": self.diagonal_minimum,
            "diagonal_maximum": self.diagonal_maximum,
            "diagonal_status": self.diagonal_status,
            "positive_edges": self.positive_edges,
            "negative_edges": self.negative_edges,
            "zero_edges": self.zero_edges,
        }


@dataclass(frozen=True)
class ConnectomeData:
    """An ingested matrix, distinct from any future graph representation."""

    matrix: NDArray[np.float64]
    roi_ids: tuple[str, ...]
    roi_labels: tuple[str | None, ...]
    coordinates: tuple[tuple[float, float, float] | None, ...]
    atlas: str | None
    input_type: InputType
    matrix_semantics: MatrixSemantics
    source_path: Path
    label_source: str
    diagnostics: MatrixDiagnostics
    number_of_observations: int | None = None
    estimator_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def number_of_rois(self) -> int:
        return int(self.matrix.shape[0])

    def inspection_dict(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type,
            "source_path": str(self.source_path),
            "number_of_rois": self.number_of_rois,
            "number_of_observations": self.number_of_observations,
            "matrix_semantics": self.matrix_semantics,
            "roi_ids": list(self.roi_ids),
            "roi_labels": list(self.roi_labels),
            "coordinates": [
                list(value) if value is not None else None for value in self.coordinates
            ],
            "label_source": self.label_source,
            "atlas": self.atlas,
            "diagnostics": self.diagnostics.to_dict(),
            "estimator_metadata": self.estimator_metadata,
            "warnings": list(self.warnings),
        }

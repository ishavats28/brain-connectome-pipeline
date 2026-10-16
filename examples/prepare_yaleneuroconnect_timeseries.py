"""Prepare one downloaded YaleNeuroConnect Shen268 ROI table for this pipeline.

The OpenNeuro derivative used in the public connectome demonstration has an observation
index column and a trailing empty column.  This utility removes only those two
non-ROI columns; it does not alter, filter, regress, or otherwise preprocess
the 268 ROI signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(input_path: Path, output_path: Path) -> None:
    """Validate and copy the 268 ROI columns without modifying the source file."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the downloaded source path.")
    try:
        frame = pd.read_csv(input_path, sep="\t")
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read the downloaded ROI table: {error}") from error
    if frame.shape[1] < 3:
        raise ValueError("Expected an observation index, ROI columns, and a trailing empty column.")

    index_name = str(frame.columns[0]).strip()
    if index_name != "ROI":
        raise ValueError("Expected the first column header to be 'ROI' for the observation index.")
    index_values = frame.iloc[:, 0].to_numpy()
    if list(index_values) != list(range(len(frame))):
        raise ValueError("Expected the first column to be a zero-based observation index.")

    trailing_name = str(frame.columns[-1]).strip()
    if not trailing_name.startswith("Unnamed:") or not frame.iloc[:, -1].isna().all():
        raise ValueError(
            "Expected the final field to have an empty header and contain only missing values."
        )

    roi_frame = frame.iloc[:, 1:-1].copy()
    roi_frame.columns = [str(column).strip() for column in roi_frame.columns]
    if roi_frame.shape[1] != 268:
        raise ValueError(f"Expected 268 Shen ROI columns; found {roi_frame.shape[1]}.")
    if not roi_frame.columns.is_unique or not all(roi_frame.columns):
        raise ValueError("ROI identifiers must be present and unique.")
    try:
        roi_frame = roi_frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Every retained ROI signal value must be numeric.") from error
    if not np.isfinite(roi_frame.to_numpy(dtype=np.float64)).all():
        raise ValueError("ROI signal values must be finite and nonmissing.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roi_frame.to_csv(output_path, sep="\t", index=False, float_format="%.17g")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

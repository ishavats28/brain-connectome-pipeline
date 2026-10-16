"""Regression tests for the documented YaleNeuroConnect table preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.prepare_yaleneuroconnect_timeseries import prepare


def _source_table(path: Path, *, observations: int = 3) -> list[str]:
    roi_ids = [f"ROI {index}" for index in range(268)]
    frame = pd.DataFrame(
        np.arange(observations * 268, dtype=float).reshape(observations, 268),
        columns=roi_ids,
    )
    frame.insert(0, "ROI", range(observations))
    frame["Unnamed: 269"] = np.nan
    frame.to_csv(path, sep="\t", index=False)
    return roi_ids


def test_preparation_preserves_source_roi_order_and_numeric_values(tmp_path: Path) -> None:
    source = tmp_path / "downloaded.tsv"
    output = tmp_path / "cleaned.tsv"
    roi_ids = _source_table(source)
    original_bytes = source.read_bytes()

    prepare(source, output)

    cleaned = pd.read_csv(output, sep="\t")
    assert source.read_bytes() == original_bytes
    assert cleaned.columns.tolist() == roi_ids
    assert cleaned.shape == (3, 268)
    assert np.isfinite(cleaned.to_numpy(dtype=float)).all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.rename(columns={"ROI": "ROI 268"}), "first column header"),
        (
            lambda frame: frame.rename(columns={"Unnamed: 269": "ROI 268"}),
            "final field",
        ),
    ],
)
def test_preparation_does_not_drop_columns_without_exact_schema(
    tmp_path: Path, mutation, message: str
) -> None:
    source = tmp_path / "downloaded.tsv"
    _source_table(source)
    frame = pd.read_csv(source, sep="\t")
    mutation(frame).to_csv(source, sep="\t", index=False)

    with pytest.raises(ValueError, match=message):
        prepare(source, tmp_path / "cleaned.tsv")


@pytest.mark.parametrize("invalid_value", ["not-a-number", np.inf])
def test_preparation_rejects_nonnumeric_or_nonfinite_roi_values(
    tmp_path: Path, invalid_value: object
) -> None:
    source = tmp_path / "downloaded.tsv"
    _source_table(source)
    frame = pd.read_csv(source, sep="\t")
    if isinstance(invalid_value, str):
        frame["ROI 12"] = frame["ROI 12"].astype(object)
    frame.loc[0, "ROI 12"] = invalid_value
    frame.to_csv(source, sep="\t", index=False)

    with pytest.raises(ValueError, match="numeric|finite"):
        prepare(source, tmp_path / "cleaned.tsv")


def test_preparation_refuses_to_overwrite_downloaded_source(tmp_path: Path) -> None:
    source = tmp_path / "downloaded.tsv"
    _source_table(source)
    original_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="must differ"):
        prepare(source, source)

    assert source.read_bytes() == original_bytes

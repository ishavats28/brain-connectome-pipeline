from pathlib import Path

import numpy as np
import pytest

from brain_connectome_pipeline.config import load_config
from brain_connectome_pipeline.connectivity import load_timeseries_connectome
from brain_connectome_pipeline.io import InputValidationError


def test_known_correlated_and_anticorrelated_series(tmp_path: Path, config_writer) -> None:
    path = tmp_path / "series.csv"
    path.write_text("A,B,C\n-2,-4,4\n-1,-2,2\n0,0,0\n1,2,-2\n2,4,-4\n", encoding="utf-8")
    result = load_timeseries_connectome(load_config(config_writer(path, source_type="timeseries")))
    expected = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    np.testing.assert_allclose(result.matrix, expected, atol=1e-15)
    assert result.number_of_observations == 5
    assert result.roi_ids == ("A", "B", "C")
    assert result.estimator_metadata["fisher_z_transformed"] is False
    assert "no confound control" in result.estimator_metadata["preprocessing"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("A,B\n1,2\n2,4\n", "at least 3"),
        ("A,B\n1,0\n1,1\n1,2\n", "Constant"),
        ("A,B\n1,0\n2,NaN\n3,2\n", "nonfinite"),
        ("A,A\n1,2\n2,3\n3,4\n", "unique"),
        ("A,B,C\n1,2\n2,3\n3,4\n", "nonfinite"),
    ],
)
def test_invalid_time_series(tmp_path: Path, config_writer, text: str, message: str) -> None:
    path = tmp_path / "series.csv"
    path.write_text(text, encoding="utf-8")
    config = load_config(config_writer(path, source_type="timeseries"))
    with pytest.raises(InputValidationError, match=message):
        load_timeseries_connectome(config)


def test_metadata_must_match_timeseries_column_order(tmp_path: Path, config_writer) -> None:
    path = tmp_path / "series.tsv"
    path.write_text("A\tB\n1\t3\n2\t5\n4\t6\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("roi_id,label\nB,Beta\nA,Alpha\n", encoding="utf-8")
    config = load_config(
        config_writer(
            path,
            source_type="timeseries",
            extra_input=f"  labels_path: {labels.as_posix()}\n",
        )
    )
    with pytest.raises(InputValidationError, match="exactly match"):
        load_timeseries_connectome(config)

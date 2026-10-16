import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain_connectome_pipeline.config import load_config
from brain_connectome_pipeline.pipeline import PipelineExecutionError, run_pipeline


def _correlation_matrix(size: int) -> np.ndarray:
    matrix = np.eye(size)
    for source in range(size):
        for target in range(source + 1, size):
            value = 0.9 - 0.06 * (source + target)
            matrix[source, target] = matrix[target, source] = value
    return matrix


def test_unified_pipeline_writes_manifest_report_and_figures(tmp_path: Path, config_writer) -> None:
    identifiers = [f"ROI_{index:02d}" for index in range(6)]
    matrix_path = tmp_path / "connectome.csv"
    pd.DataFrame(_correlation_matrix(6), index=identifiers, columns=identifiers).to_csv(matrix_path)
    config = load_config(
        config_writer(
            matrix_path,
            extra_input="  embedded_labels: true\n",
            extra_root=(
                "graph:\n  enabled: true\n  mode: proportional\n  target_density: 0.4\n"
                "topology:\n  enabled: true\n"
                "community:\n  enabled: true\n"
                "smallworld:\n  enabled: false\n"
                "density_sensitivity:\n  enabled: true\n"
                "  densities: [0.2, 0.4]\n  plots_enabled: false\n"
                "robustness:\n  enabled: true\n"
                "  removal_fractions: [0.0, 0.5]\n  random_trials: 2\n  plots_enabled: false\n"
                "reporting:\n  enabled: true\n"
            ),
        )
    )
    source_before = matrix_path.read_text(encoding="utf-8")
    result = run_pipeline(config, output_dir=tmp_path / "run")
    assert result.stage_status["graph"]["status"] == "complete"
    assert result.stage_status["smallworld"]["status"] == "skipped"
    assert (result.output_root / "execution_manifest.json").is_file()
    assert (result.output_root / "report.html").is_file()
    assert (result.output_root / "graph" / "graph_diagnostics.json").is_file()
    assert (result.output_root / "topology" / "global_metrics.tsv").is_file()
    assert (result.output_root / "communities" / "community_assignments.tsv").is_file()
    assert (result.output_root / "sensitivity" / "density_sensitivity.tsv").is_file()
    assert (result.output_root / "robustness" / "robustness_summary.json").is_file()
    assert (result.output_root / "figures" / "connectivity_matrix.png").stat().st_size > 0
    report = (result.output_root / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in report
    manifest = json.loads((result.output_root / "execution_manifest.json").read_text())
    provenance = json.loads((result.output_root / "provenance.json").read_text())
    effective_config = json.loads((result.output_root / "effective_config.json").read_text())
    assert manifest["completion_status"] == "complete"
    assert provenance["completion_status"] == "complete"
    assert provenance["input"]["roi_identifiers"] == identifiers
    assert effective_config["input"]["path"] == matrix_path.name
    assert not Path(effective_config["input"]["path"]).is_absolute()
    assert all(not Path(path).is_absolute() for path in manifest["output_files"].values())
    assert matrix_path.read_text(encoding="utf-8") == source_before
    with pytest.raises(PipelineExecutionError, match="Refusing to overwrite"):
        run_pipeline(config, output_dir=result.output_root)
    unrelated = result.output_root / "research_notes.txt"
    unrelated.write_text("preserve me\n", encoding="utf-8")
    run_pipeline(config, output_dir=result.output_root, overwrite=True)
    assert unrelated.read_text(encoding="utf-8") == "preserve me\n"


def test_unified_pipeline_accepts_timeseries_and_can_skip_optional_stages(
    tmp_path: Path, config_writer
) -> None:
    timeseries_path = tmp_path / "timeseries.tsv"
    pd.DataFrame(
        {
            "ROI_A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "ROI_B": [0.0, 2.0, 4.0, 6.0, 8.0],
            "ROI_C": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    ).to_csv(timeseries_path, sep="\t", index=False)
    config = load_config(
        config_writer(
            timeseries_path,
            source_type="timeseries",
            extra_root=(
                "graph:\n  enabled: true\n"
                "topology:\n  enabled: false\n"
                "community:\n  enabled: false\n"
                "smallworld:\n  enabled: false\n"
                "density_sensitivity:\n  enabled: false\n"
                "robustness:\n  enabled: false\n"
                "reporting:\n  enabled: false\n"
            ),
        )
    )
    result = run_pipeline(config, output_dir=tmp_path / "timeseries_run")
    assert result.stage_status["topology"]["status"] == "skipped"
    assert result.stage_status["report"]["status"] == "skipped"
    assert not (result.output_root / "report.html").exists()
    provenance = json.loads((result.output_root / "provenance.json").read_text())
    assert provenance["input"]["source_type"] == "timeseries"
    assert provenance["input"]["number_of_observations"] == 5


def test_required_stage_failure_never_writes_completion_manifest(
    tmp_path: Path, config_writer, monkeypatch
) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0.8\n0.8,1\n", encoding="utf-8")
    config = load_config(
        config_writer(
            matrix,
            extra_root=(
                "graph:\n  enabled: true\n"
                "topology:\n  enabled: false\n"
                "reporting:\n  enabled: false\n"
            ),
        )
    )

    def fail_figures(*args, **kwargs):
        raise OSError("synthetic figure failure")

    monkeypatch.setattr("brain_connectome_pipeline.pipeline.write_primary_figures", fail_figures)
    output = tmp_path / "failed_run"
    with pytest.raises(PipelineExecutionError, match="synthetic figure failure"):
        run_pipeline(config, output_dir=output)
    assert not (output / "execution_manifest.json").exists()

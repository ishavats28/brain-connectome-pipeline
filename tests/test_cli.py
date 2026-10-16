from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from brain_connectome_pipeline import __version__
from brain_connectome_pipeline.cli import main


def test_cli_help(capsys) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "inspect-input" in output
    assert "build-graph" in output
    assert "analyze-topology" in output
    assert "analyze-communities" in output
    assert "analyze-smallworld" in output
    assert "analyze-sensitivity" in output
    assert "analyze-robustness" in output
    assert "run" in output


def test_cli_version(capsys) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_validate_and_inspect_json(tmp_path: Path, config_writer, capsys) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,-0.2\n-0.2,1\n", encoding="utf-8")
    config = config_writer(matrix)
    assert main(["validate-config", "--config", str(config)]) == 0
    assert "Configuration valid" in capsys.readouterr().out
    assert main(["inspect-input", "--config", str(config), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"negative_edges": 1' in output
    assert '"number_of_rois": 2' in output


def test_cli_failure_has_nonzero_exit(tmp_path: Path, config_writer, capsys) -> None:
    config = config_writer(tmp_path / "missing.csv")
    assert main(["inspect-input", "--config", str(config)]) == 2
    assert "error:" in capsys.readouterr().err


def test_build_graph_cli_success_and_collision(tmp_path: Path, config_writer, capsys) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0.8,-0.2\n0.8,1,0.6\n-0.2,0.6,1\n", encoding="utf-8")
    config = config_writer(
        matrix,
        extra_root=("graph:\n  enabled: true\n  mode: proportional\n  target_density: 0.34\n"),
    )
    output_dir = tmp_path / "graph_outputs"
    arguments = [
        "build-graph",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert '"retained_edges": 1' in output
    assert (output_dir / "graph_diagnostics.json").is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_analyze_topology_cli_outputs_and_collision(tmp_path: Path, config_writer, capsys) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,1,0\n1,1,1\n0,1,1\n", encoding="utf-8")
    config = config_writer(
        matrix,
        extra_root=("graph:\n  enabled: true\n  mode: unthresholded\ntopology:\n  enabled: true\n"),
    )
    output_dir = tmp_path / "topology_outputs"
    arguments = [
        "analyze-topology",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert '"global_metrics_available"' in output
    assert (output_dir / "global_metrics.tsv").is_file()
    assert (output_dir / "nodal_metrics.tsv").is_file()
    assert (output_dir / "topology_results.json").is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_analyze_communities_cli_outputs_and_collision(
    tmp_path: Path, config_writer, capsys
) -> None:
    matrix = np.eye(6)
    for group in ((0, 1, 2), (3, 4, 5)):
        for source in group:
            for target in group:
                if source < target:
                    matrix[source, target] = matrix[target, source] = 0.9
    matrix[2, 3] = matrix[3, 2] = 0.1
    matrix_path = tmp_path / "communities.csv"
    pd.DataFrame(matrix).to_csv(matrix_path, header=False, index=False)
    config = config_writer(
        matrix_path,
        extra_root=(
            "graph:\n  enabled: true\n  mode: unthresholded\n"
            "topology:\n  enabled: true\n"
            "community:\n"
            "  enabled: true\n"
            "  seed: 13\n"
            "  hub_percentile_threshold: 0.75\n"
        ),
    )
    output_dir = tmp_path / "community_outputs"
    arguments = [
        "analyze-communities",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert '"number_of_communities": 2' in output
    for name in (
        "community_assignments.tsv",
        "community_summary.json",
        "participation_metrics.tsv",
        "hub_candidates.tsv",
        "hub_summary.json",
    ):
        assert (output_dir / name).is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_analyze_smallworld_cli_outputs_and_collision(
    tmp_path: Path, config_writer, capsys
) -> None:
    graph = nx.watts_strogatz_graph(12, 4, 0.25, seed=4)
    matrix = nx.to_numpy_array(graph)
    np.fill_diagonal(matrix, 1.0)
    matrix_path = tmp_path / "smallworld.csv"
    pd.DataFrame(matrix).to_csv(matrix_path, header=False, index=False)
    config = config_writer(
        matrix_path,
        extra_root=(
            "graph:\n  enabled: true\n  mode: unthresholded\n"
            "smallworld:\n"
            "  enabled: true\n"
            "  ensemble_size: 3\n"
            "  minimum_valid_samples: 2\n"
            "  swaps_per_edge: 1\n"
            "  maximum_ensemble_attempts: 12\n"
            "  seed: 41\n"
        ),
    )
    output_dir = tmp_path / "smallworld_outputs"
    arguments = [
        "analyze-smallworld",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert '"status": "available"' in output
    assert (output_dir / "smallworld_summary.json").is_file()
    assert (output_dir / "smallworld_nulls.tsv").is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_analyze_sensitivity_cli_outputs_and_collision(
    tmp_path: Path, config_writer, capsys
) -> None:
    matrix = np.eye(5)
    for source in range(5):
        for target in range(source + 1, 5):
            matrix[source, target] = matrix[target, source] = 0.9 - 0.05 * (source + target)
    matrix_path = tmp_path / "sensitivity.csv"
    pd.DataFrame(matrix).to_csv(matrix_path, header=False, index=False)
    config = config_writer(
        matrix_path,
        extra_root=(
            "graph:\n  enabled: true\n"
            "density_sensitivity:\n"
            "  enabled: true\n"
            "  densities: [0.1, 0.2]\n"
            "  plots_enabled: false\n"
        ),
    )
    output_dir = tmp_path / "sensitivity_outputs"
    arguments = [
        "analyze-sensitivity",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert '"density_points": 2' in output
    assert (output_dir / "density_sensitivity.tsv").is_file()
    assert (output_dir / "density_sensitivity_summary.json").is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_analyze_robustness_cli_outputs_and_collision(
    tmp_path: Path, config_writer, capsys
) -> None:
    matrix = np.eye(5)
    for leaf in range(1, 5):
        matrix[0, leaf] = matrix[leaf, 0] = 0.8
    matrix_path = tmp_path / "robustness.csv"
    pd.DataFrame(matrix).to_csv(matrix_path, header=False, index=False)
    config = config_writer(
        matrix_path,
        extra_root=(
            "graph:\n  enabled: true\n"
            "robustness:\n"
            "  enabled: true\n"
            "  removal_fractions: [0.0, 0.2, 0.4]\n"
            "  random_trials: 3\n"
            "  seed: 5\n"
            "  plots_enabled: false\n"
        ),
    )
    output_dir = tmp_path / "robustness_outputs"
    arguments = [
        "analyze-robustness",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert '"status": "available"' in output
    for name in (
        "robustness_trials.tsv",
        "robustness_summary.tsv",
        "robustness_ranking.tsv",
        "robustness_summary.json",
    ):
        assert (output_dir / name).is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert main([*arguments, "--overwrite"]) == 0


def test_unified_run_cli_json(tmp_path: Path, config_writer, capsys) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0.8,0.4\n0.8,1,0.6\n0.4,0.6,1\n", encoding="utf-8")
    config = config_writer(
        matrix,
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
    output_dir = tmp_path / "unified"
    arguments = ["run", "--config", str(config), "--output-dir", str(output_dir), "--json"]
    assert main(arguments) == 0
    assert '"status": "complete"' in capsys.readouterr().out
    assert (output_dir / "execution_manifest.json").is_file()
    assert main(arguments) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err

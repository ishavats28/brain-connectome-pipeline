from pathlib import Path

import pytest

from brain_connectome_pipeline.config import ConfigurationError, load_config


def test_valid_yaml_resolves_paths(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    config = load_config(config_writer(matrix))
    assert config.input.path == matrix.resolve()
    assert config.random_seed == 1729


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("input: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_config(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("surprise: true\n", "Unknown key"),
        ("input:\n  source_type: matrix\n", "input.path"),
        (
            "input:\n  source_type: volume\n  path: x.csv\n  matrix_semantics: correlation\n",
            "source_type",
        ),
        (
            "input:\n  source_type: matrix\n  path: x.csv\n  matrix_semantics: mystery\n",
            "matrix_semantics",
        ),
    ],
)
def test_invalid_configuration_keys_and_values(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(path, require_input_exists=False)


def test_required_input_path_must_exist(tmp_path: Path, config_writer) -> None:
    with pytest.raises(ConfigurationError, match="Input file not found"):
        load_config(config_writer(tmp_path / "absent.csv"))


def test_removed_null_models_section_is_rejected_as_unknown(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unknown key.*null_models"):
        load_config(config_writer(matrix, extra_root="null_models:\n  enabled: true\n"))


def test_reporting_configuration_is_available(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    config = load_config(config_writer(matrix, extra_root="reporting:\n  enabled: false\n"))
    assert not config.reporting.enabled


def test_atlas_must_be_a_nonempty_string(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="input.atlas"):
        load_config(config_writer(matrix, extra_input="  atlas: 123\n"))


def test_valid_graph_configuration(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    config = load_config(
        config_writer(
            matrix,
            extra_root=(
                "graph:\n"
                "  enabled: true\n"
                "  mode: proportional\n"
                "  target_density: 0.25\n"
                "  tie_handling: roi_index\n"
            ),
        )
    )
    assert config.graph.enabled
    assert config.graph.target_density == 0.25


@pytest.mark.parametrize(
    ("graph_yaml", "message"),
    [
        ("  mode: proportional\n", "target_density"),
        ("  mode: proportional\n  target_density: 1.1\n", "between 0 and 1"),
        ("  mode: unthresholded\n  target_density: 0.2\n", "only valid"),
        ("  positive_edge_policy: absolute\n", "strictly_positive"),
        ("  tie_handling: random\n", "roi_index"),
    ],
)
def test_invalid_graph_configuration(
    tmp_path: Path, config_writer, graph_yaml: str, message: str
) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(config_writer(matrix, extra_root="graph:\n" + graph_yaml))


def test_valid_topology_configuration(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    config = load_config(
        config_writer(
            matrix,
            extra_root=(
                "topology:\n"
                "  enabled: true\n"
                "  report_largest_component_paths: false\n"
                "  eigenvector_max_iterations: 250\n"
                "  eigenvector_tolerance: 1.0e-7\n"
            ),
        )
    )
    assert config.topology.enabled
    assert not config.topology.report_largest_component_paths
    assert config.topology.eigenvector_max_iterations == 250


@pytest.mark.parametrize(
    ("topology_yaml", "message"),
    [
        ("  eigenvector_max_iterations: 0\n", "positive integer"),
        ("  eigenvector_tolerance: 0\n", "positive finite"),
        ("  report_largest_component_paths: yes-please\n", "true or false"),
        ("  mystery: 1\n", "Unknown key"),
    ],
)
def test_invalid_topology_configuration(
    tmp_path: Path, config_writer, topology_yaml: str, message: str
) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,0\n0,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(config_writer(matrix, extra_root="topology:\n" + topology_yaml))


def test_valid_community_and_smallworld_configuration(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,1\n1,1\n", encoding="utf-8")
    config = load_config(
        config_writer(
            matrix,
            extra_root=(
                "graph:\n  enabled: true\n"
                "topology:\n  enabled: true\n"
                "community:\n"
                "  enabled: true\n"
                "  resolution: 1.25\n"
                "  seed: 7\n"
                "  hub_percentile_threshold: 0.75\n"
                "smallworld:\n"
                "  enabled: true\n"
                "  ensemble_size: 4\n"
                "  minimum_valid_samples: 2\n"
                "  swaps_per_edge: 2\n"
                "  maximum_ensemble_attempts: 8\n"
                "  seed: 11\n"
            ),
        )
    )
    assert config.community.enabled
    assert config.community.resolution == 1.25
    assert config.community.seed == 7
    assert config.smallworld.enabled
    assert config.smallworld.minimum_valid_samples == 2


@pytest.mark.parametrize(
    ("extra_root", "message"),
    [
        ("community:\n  resolution: 0\n", "positive finite"),
        ("community:\n  hub_percentile_threshold: 0\n", "greater than 0"),
        ("community:\n  hub_minimum_metrics: 5\n", "between 1 and 4"),
        (
            "community:\n  enabled: true\n  hub_enabled: true\n",
            "topology.enabled",
        ),
        (
            "topology:\n  enabled: true\ncommunity:\n"
            "  enabled: true\n  participation_enabled: false\n  hub_enabled: true\n",
            "participation_enabled",
        ),
        (
            "smallworld:\n  ensemble_size: 2\n  minimum_valid_samples: 3\n",
            "cannot exceed",
        ),
        (
            "smallworld:\n  ensemble_size: 3\n  minimum_valid_samples: 3\n"
            "  maximum_ensemble_attempts: 2\n",
            "cannot be smaller",
        ),
        ("smallworld:\n  swaps_per_edge: 0\n", "positive integer"),
    ],
)
def test_invalid_community_and_smallworld_configuration(
    tmp_path: Path, config_writer, extra_root: str, message: str
) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,1\n1,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(config_writer(matrix, extra_root=extra_root))


def test_valid_sensitivity_and_robustness_configuration(tmp_path: Path, config_writer) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,1\n1,1\n", encoding="utf-8")
    config = load_config(
        config_writer(
            matrix,
            extra_root=(
                "graph:\n  enabled: true\n"
                "density_sensitivity:\n"
                "  enabled: true\n"
                "  densities: [0.1, 0.2, 0.4]\n"
                "  plots_enabled: false\n"
                "robustness:\n"
                "  enabled: true\n"
                "  ranking_criterion: degree\n"
                "  removal_fractions: [0.0, 0.25, 0.5]\n"
                "  random_trials: 7\n"
                "  seed: 19\n"
                "  plots_enabled: false\n"
            ),
        )
    )
    assert config.density_sensitivity.enabled
    assert config.density_sensitivity.densities == (0.1, 0.2, 0.4)
    assert not config.density_sensitivity.plots_enabled
    assert config.robustness.enabled
    assert config.robustness.ranking_criterion == "degree"
    assert config.robustness.random_trials == 7


@pytest.mark.parametrize(
    ("extra_root", "message"),
    [
        (
            "density_sensitivity:\n  densities: [0.1, 0.1]\n",
            "strictly increasing",
        ),
        (
            "density_sensitivity:\n  densities: [0.2, 0.1]\n",
            "strictly increasing",
        ),
        ("density_sensitivity:\n  densities: [0.0, 0.1]\n", "greater than"),
        ("robustness:\n  removal_fractions: [0.0, 0.2, 0.2]\n", "strictly increasing"),
        ("robustness:\n  removal_fractions: [0.1, 0.2]\n", "begin with 0.0"),
        ("robustness:\n  ranking_criterion: eigenvector\n", "strength.*degree"),
        ("robustness:\n  random_trials: 0\n", "between 1 and 1000"),
        ("robustness:\n  random_trials: 1001\n", "between 1 and 1000"),
        (
            "density_sensitivity:\n  enabled: true\n",
            "graph.enabled",
        ),
    ],
)
def test_invalid_sensitivity_and_robustness_configuration(
    tmp_path: Path, config_writer, extra_root: str, message: str
) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text("1,1\n1,1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(config_writer(matrix, extra_root=extra_root))

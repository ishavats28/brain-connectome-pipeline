"""Strict YAML configuration loading for the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from brain_connectome_pipeline.models import MatrixSemantics


class ConfigurationError(ValueError):
    """Raised when configuration is missing, malformed, or unsupported."""


@dataclass(frozen=True)
class InputConfig:
    source_type: Literal["matrix", "timeseries"]
    path: Path
    matrix_semantics: MatrixSemantics
    labels_path: Path | None = None
    embedded_labels: bool = False
    atlas: str | None = None
    symmetry_tolerance: float = 1e-8
    correlation_tolerance: float = 1e-8


@dataclass(frozen=True)
class ConnectivityConfig:
    estimator: Literal["pearson"] = "pearson"
    orientation: Literal["observations_by_rois"] = "observations_by_rois"
    minimum_observations: int = 3
    constant_tolerance: float = 1e-12


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = False
    mode: Literal["unthresholded", "proportional"] = "unthresholded"
    target_density: float | None = None
    positive_edge_policy: Literal["strictly_positive"] = "strictly_positive"
    tie_handling: Literal["roi_index"] = "roi_index"
    allow_binary_subsampling: bool = False
    write_binary: bool = True
    write_weighted: bool = True


@dataclass(frozen=True)
class TopologyConfig:
    enabled: bool = False
    report_largest_component_paths: bool = True
    eigenvector_max_iterations: int = 1000
    eigenvector_tolerance: float = 1e-6


@dataclass(frozen=True)
class CommunityConfig:
    enabled: bool = False
    resolution: float = 1.0
    seed: int = 1729
    participation_enabled: bool = True
    hub_enabled: bool = True
    hub_percentile_threshold: float = 0.80
    hub_minimum_metrics: int = 3


@dataclass(frozen=True)
class SmallWorldConfig:
    enabled: bool = False
    ensemble_size: int = 10
    minimum_valid_samples: int = 5
    swaps_per_edge: int = 5
    maximum_ensemble_attempts: int = 30
    seed: int = 1729


@dataclass(frozen=True)
class DensitySensitivityConfig:
    enabled: bool = False
    densities: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
    plots_enabled: bool = True


@dataclass(frozen=True)
class RobustnessConfig:
    enabled: bool = False
    ranking_criterion: Literal["strength", "degree"] = "strength"
    removal_fractions: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    random_trials: int = 20
    seed: int = 1729
    plots_enabled: bool = True


@dataclass(frozen=True)
class ReportingConfig:
    """Controls generation of the offline HTML report during a unified run."""

    enabled: bool = True


@dataclass(frozen=True)
class OutputConfig:
    directory: Path = Path("outputs")


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    smallworld: SmallWorldConfig = field(default_factory=SmallWorldConfig)
    density_sensitivity: DensitySensitivityConfig = field(default_factory=DensitySensitivityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int = 1729


_ROOT_KEYS = {
    "input",
    "connectivity",
    "graph",
    "topology",
    "community",
    "smallworld",
    "density_sensitivity",
    "robustness",
    "reporting",
    "outputs",
    "random_seed",
}
_INPUT_KEYS = {
    "source_type",
    "path",
    "matrix_semantics",
    "labels_path",
    "embedded_labels",
    "atlas",
    "symmetry_tolerance",
    "correlation_tolerance",
}
_CONNECTIVITY_KEYS = {"estimator", "orientation", "minimum_observations", "constant_tolerance"}
_GRAPH_KEYS = {
    "enabled",
    "mode",
    "target_density",
    "positive_edge_policy",
    "tie_handling",
    "allow_binary_subsampling",
    "write_binary",
    "write_weighted",
}
_TOPOLOGY_KEYS = {
    "enabled",
    "report_largest_component_paths",
    "eigenvector_max_iterations",
    "eigenvector_tolerance",
}
_COMMUNITY_KEYS = {
    "enabled",
    "resolution",
    "seed",
    "participation_enabled",
    "hub_enabled",
    "hub_percentile_threshold",
    "hub_minimum_metrics",
}
_SMALLWORLD_KEYS = {
    "enabled",
    "ensemble_size",
    "minimum_valid_samples",
    "swaps_per_edge",
    "maximum_ensemble_attempts",
    "seed",
}
_DENSITY_SENSITIVITY_KEYS = {"enabled", "densities", "plots_enabled"}
_ROBUSTNESS_KEYS = {
    "enabled",
    "ranking_criterion",
    "removal_fractions",
    "random_trials",
    "seed",
    "plots_enabled",
}


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"'{context}' must be a YAML mapping.")
    return value


def _known_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown key(s) in '{context}': {', '.join(unknown)}")


def _resolve_path(value: Any, base: Path, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{key}' must be a non-empty path string.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _strict_float_sequence(
    value: Any,
    *,
    key: str,
    minimum: float,
    maximum: float,
    include_minimum: bool,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigurationError(f"'{key}' must be a non-empty YAML sequence.")
    converted: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
            raise ConfigurationError(f"'{key}' values must be finite numbers.")
        number = float(item)
        minimum_valid = number >= minimum if include_minimum else number > minimum
        if not minimum_valid or number > maximum:
            boundary = "at least" if include_minimum else "greater than"
            raise ConfigurationError(
                f"'{key}' values must be {boundary} {minimum} and at most {maximum}."
            )
        converted.append(number)
    if any(left >= right for left, right in zip(converted, converted[1:], strict=False)):
        raise ConfigurationError(f"'{key}' must be strictly increasing with unique values.")
    return tuple(converted)


def load_config(path: str | Path, *, require_input_exists: bool = True) -> PipelineConfig:
    """Load a strict configuration, resolving relative paths from the YAML location."""
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    _known_keys(root, _ROOT_KEYS, "configuration")
    input_raw = _mapping(root.get("input"), "input")
    _known_keys(input_raw, _INPUT_KEYS, "input")
    for required in ("source_type", "path", "matrix_semantics"):
        if required not in input_raw:
            raise ConfigurationError(f"Missing required key 'input.{required}'.")

    source_type = input_raw["source_type"]
    if source_type not in ("matrix", "timeseries"):
        raise ConfigurationError("'input.source_type' must be 'matrix' or 'timeseries'.")
    semantics = input_raw["matrix_semantics"]
    if semantics not in get_args(MatrixSemantics):
        accepted = ", ".join(get_args(MatrixSemantics))
        raise ConfigurationError(f"'input.matrix_semantics' must be one of: {accepted}.")
    if source_type == "timeseries" and semantics != "correlation":
        raise ConfigurationError("Time-series input produces 'correlation' matrix semantics.")

    base = config_path.parent
    input_path = _resolve_path(input_raw["path"], base, "input.path")
    labels_path = input_raw.get("labels_path")
    labels = (
        _resolve_path(labels_path, base, "input.labels_path") if labels_path is not None else None
    )
    if require_input_exists and not input_path.is_file():
        raise ConfigurationError(f"Input file not found: {input_path}")
    if require_input_exists and labels is not None and not labels.is_file():
        raise ConfigurationError(f"ROI labels file not found: {labels}")

    symmetry_tolerance = input_raw.get("symmetry_tolerance", 1e-8)
    correlation_tolerance = input_raw.get("correlation_tolerance", 1e-8)
    if not isinstance(symmetry_tolerance, (int, float)) or symmetry_tolerance <= 0:
        raise ConfigurationError("'input.symmetry_tolerance' must be positive.")
    if not isinstance(correlation_tolerance, (int, float)) or correlation_tolerance <= 0:
        raise ConfigurationError("'input.correlation_tolerance' must be positive.")
    embedded = input_raw.get("embedded_labels", False)
    if not isinstance(embedded, bool):
        raise ConfigurationError("'input.embedded_labels' must be true or false.")
    if embedded and input_path.suffix.lower() not in {".csv", ".tsv"}:
        raise ConfigurationError("Embedded labels are supported only for CSV and TSV matrices.")
    if embedded and labels is not None:
        raise ConfigurationError("Use either embedded labels or labels_path, not both.")
    atlas = input_raw.get("atlas")
    if atlas is not None and (not isinstance(atlas, str) or not atlas.strip()):
        raise ConfigurationError("'input.atlas' must be a non-empty string when supplied.")

    conn_raw = _mapping(root.get("connectivity"), "connectivity")
    _known_keys(conn_raw, _CONNECTIVITY_KEYS, "connectivity")
    try:
        connectivity = ConnectivityConfig(**conn_raw)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid connectivity configuration: {exc}") from exc
    if connectivity.estimator != "pearson":
        raise ConfigurationError("Only ordinary Pearson correlation is implemented.")
    if connectivity.orientation != "observations_by_rois":
        raise ConfigurationError("Time-series orientation must be 'observations_by_rois'.")
    if connectivity.minimum_observations < 3:
        raise ConfigurationError("'connectivity.minimum_observations' must be at least 3.")
    if connectivity.constant_tolerance < 0:
        raise ConfigurationError("'connectivity.constant_tolerance' cannot be negative.")

    graph_raw = _mapping(root.get("graph"), "graph")
    _known_keys(graph_raw, _GRAPH_KEYS, "graph")
    graph_enabled = graph_raw.get("enabled", False)
    graph_mode = graph_raw.get("mode", "unthresholded")
    target_density = graph_raw.get("target_density")
    positive_edge_policy = graph_raw.get("positive_edge_policy", "strictly_positive")
    tie_handling = graph_raw.get("tie_handling", "roi_index")
    allow_binary_subsampling = graph_raw.get("allow_binary_subsampling", False)
    write_binary = graph_raw.get("write_binary", True)
    write_weighted = graph_raw.get("write_weighted", True)
    for key, value in {
        "enabled": graph_enabled,
        "allow_binary_subsampling": allow_binary_subsampling,
        "write_binary": write_binary,
        "write_weighted": write_weighted,
    }.items():
        if not isinstance(value, bool):
            raise ConfigurationError(f"'graph.{key}' must be true or false.")
    if graph_mode not in ("unthresholded", "proportional"):
        raise ConfigurationError("'graph.mode' must be 'unthresholded' or 'proportional'.")
    if positive_edge_policy != "strictly_positive":
        raise ConfigurationError("'graph.positive_edge_policy' must be 'strictly_positive'.")
    if tie_handling != "roi_index":
        raise ConfigurationError("'graph.tie_handling' must be 'roi_index'.")
    if graph_mode == "proportional":
        if (
            not isinstance(target_density, (int, float))
            or isinstance(target_density, bool)
            or not 0 <= target_density <= 1
        ):
            raise ConfigurationError(
                "'graph.target_density' must be a number between 0 and 1 in proportional mode."
            )
        target_density = float(target_density)
    elif target_density is not None:
        raise ConfigurationError(
            "'graph.target_density' is only valid when graph.mode is 'proportional'."
        )
    graph = GraphConfig(
        enabled=graph_enabled,
        mode=graph_mode,
        target_density=target_density,
        positive_edge_policy=positive_edge_policy,
        tie_handling=tie_handling,
        allow_binary_subsampling=allow_binary_subsampling,
        write_binary=write_binary,
        write_weighted=write_weighted,
    )

    topology_raw = _mapping(root.get("topology"), "topology")
    _known_keys(topology_raw, _TOPOLOGY_KEYS, "topology")
    topology_enabled = topology_raw.get("enabled", False)
    report_component_paths = topology_raw.get("report_largest_component_paths", True)
    eigenvector_iterations = topology_raw.get("eigenvector_max_iterations", 1000)
    eigenvector_tolerance = topology_raw.get("eigenvector_tolerance", 1e-6)
    for key, value in {
        "enabled": topology_enabled,
        "report_largest_component_paths": report_component_paths,
    }.items():
        if not isinstance(value, bool):
            raise ConfigurationError(f"'topology.{key}' must be true or false.")
    if (
        not isinstance(eigenvector_iterations, int)
        or isinstance(eigenvector_iterations, bool)
        or eigenvector_iterations < 1
    ):
        raise ConfigurationError(
            "'topology.eigenvector_max_iterations' must be a positive integer."
        )
    if (
        not isinstance(eigenvector_tolerance, (int, float))
        or isinstance(eigenvector_tolerance, bool)
        or not math.isfinite(eigenvector_tolerance)
        or eigenvector_tolerance <= 0
    ):
        raise ConfigurationError(
            "'topology.eigenvector_tolerance' must be a positive finite number."
        )
    topology = TopologyConfig(
        enabled=topology_enabled,
        report_largest_component_paths=report_component_paths,
        eigenvector_max_iterations=eigenvector_iterations,
        eigenvector_tolerance=float(eigenvector_tolerance),
    )

    community_raw = _mapping(root.get("community"), "community")
    _known_keys(community_raw, _COMMUNITY_KEYS, "community")
    community_enabled = community_raw.get("enabled", False)
    community_resolution = community_raw.get("resolution", 1.0)
    community_seed = community_raw.get("seed", 1729)
    participation_enabled = community_raw.get("participation_enabled", True)
    hub_enabled = community_raw.get("hub_enabled", True)
    hub_threshold = community_raw.get("hub_percentile_threshold", 0.80)
    hub_minimum_metrics = community_raw.get("hub_minimum_metrics", 3)
    for key, value in {
        "enabled": community_enabled,
        "participation_enabled": participation_enabled,
        "hub_enabled": hub_enabled,
    }.items():
        if not isinstance(value, bool):
            raise ConfigurationError(f"'community.{key}' must be true or false.")
    if (
        not isinstance(community_resolution, (int, float))
        or isinstance(community_resolution, bool)
        or not math.isfinite(community_resolution)
        or community_resolution <= 0
    ):
        raise ConfigurationError("'community.resolution' must be a positive finite number.")
    if (
        not isinstance(community_seed, int)
        or isinstance(community_seed, bool)
        or community_seed < 0
    ):
        raise ConfigurationError("'community.seed' must be a nonnegative integer.")
    if (
        not isinstance(hub_threshold, (int, float))
        or isinstance(hub_threshold, bool)
        or not math.isfinite(hub_threshold)
        or not 0 < hub_threshold <= 1
    ):
        raise ConfigurationError(
            "'community.hub_percentile_threshold' must be greater than 0 and at most 1."
        )
    if (
        not isinstance(hub_minimum_metrics, int)
        or isinstance(hub_minimum_metrics, bool)
        or not 1 <= hub_minimum_metrics <= 4
    ):
        raise ConfigurationError("'community.hub_minimum_metrics' must be between 1 and 4.")
    if community_enabled and hub_enabled and not topology.enabled:
        raise ConfigurationError(
            "'topology.enabled' must be true when community hub characterization is enabled."
        )
    if community_enabled and hub_enabled and not participation_enabled:
        raise ConfigurationError(
            "'community.participation_enabled' must be true when hub characterization is enabled."
        )
    community = CommunityConfig(
        enabled=community_enabled,
        resolution=float(community_resolution),
        seed=community_seed,
        participation_enabled=participation_enabled,
        hub_enabled=hub_enabled,
        hub_percentile_threshold=float(hub_threshold),
        hub_minimum_metrics=hub_minimum_metrics,
    )

    smallworld_raw = _mapping(root.get("smallworld"), "smallworld")
    _known_keys(smallworld_raw, _SMALLWORLD_KEYS, "smallworld")
    smallworld_enabled = smallworld_raw.get("enabled", False)
    ensemble_size = smallworld_raw.get("ensemble_size", 10)
    minimum_valid = smallworld_raw.get("minimum_valid_samples", 5)
    swaps_per_edge = smallworld_raw.get("swaps_per_edge", 5)
    maximum_attempts = smallworld_raw.get("maximum_ensemble_attempts", 30)
    smallworld_seed = smallworld_raw.get("seed", 1729)
    if not isinstance(smallworld_enabled, bool):
        raise ConfigurationError("'smallworld.enabled' must be true or false.")
    for key, value, minimum in (
        ("ensemble_size", ensemble_size, 1),
        ("minimum_valid_samples", minimum_valid, 1),
        ("swaps_per_edge", swaps_per_edge, 1),
        ("maximum_ensemble_attempts", maximum_attempts, 1),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ConfigurationError(f"'smallworld.{key}' must be a positive integer.")
    if minimum_valid > ensemble_size:
        raise ConfigurationError("'smallworld.minimum_valid_samples' cannot exceed ensemble_size.")
    if maximum_attempts < ensemble_size:
        raise ConfigurationError(
            "'smallworld.maximum_ensemble_attempts' cannot be smaller than ensemble_size."
        )
    if (
        not isinstance(smallworld_seed, int)
        or isinstance(smallworld_seed, bool)
        or smallworld_seed < 0
    ):
        raise ConfigurationError("'smallworld.seed' must be a nonnegative integer.")
    smallworld = SmallWorldConfig(
        enabled=smallworld_enabled,
        ensemble_size=ensemble_size,
        minimum_valid_samples=minimum_valid,
        swaps_per_edge=swaps_per_edge,
        maximum_ensemble_attempts=maximum_attempts,
        seed=smallworld_seed,
    )

    sensitivity_raw = _mapping(root.get("density_sensitivity"), "density_sensitivity")
    _known_keys(sensitivity_raw, _DENSITY_SENSITIVITY_KEYS, "density_sensitivity")
    sensitivity_enabled = sensitivity_raw.get("enabled", False)
    sensitivity_plots = sensitivity_raw.get("plots_enabled", True)
    for key, value in {
        "enabled": sensitivity_enabled,
        "plots_enabled": sensitivity_plots,
    }.items():
        if not isinstance(value, bool):
            raise ConfigurationError(f"'density_sensitivity.{key}' must be true or false.")
    densities = _strict_float_sequence(
        sensitivity_raw.get("densities", [0.10, 0.15, 0.20, 0.25, 0.30]),
        key="density_sensitivity.densities",
        minimum=0.0,
        maximum=1.0,
        include_minimum=False,
    )
    density_sensitivity = DensitySensitivityConfig(
        enabled=sensitivity_enabled,
        densities=densities,
        plots_enabled=sensitivity_plots,
    )

    robustness_raw = _mapping(root.get("robustness"), "robustness")
    _known_keys(robustness_raw, _ROBUSTNESS_KEYS, "robustness")
    robustness_enabled = robustness_raw.get("enabled", False)
    ranking_criterion = robustness_raw.get("ranking_criterion", "strength")
    random_trials = robustness_raw.get("random_trials", 20)
    robustness_seed = robustness_raw.get("seed", 1729)
    robustness_plots = robustness_raw.get("plots_enabled", True)
    for key, value in {
        "enabled": robustness_enabled,
        "plots_enabled": robustness_plots,
    }.items():
        if not isinstance(value, bool):
            raise ConfigurationError(f"'robustness.{key}' must be true or false.")
    if ranking_criterion not in ("strength", "degree"):
        raise ConfigurationError("'robustness.ranking_criterion' must be 'strength' or 'degree'.")
    if (
        not isinstance(random_trials, int)
        or isinstance(random_trials, bool)
        or not 1 <= random_trials <= 1000
    ):
        raise ConfigurationError("'robustness.random_trials' must be between 1 and 1000.")
    if (
        not isinstance(robustness_seed, int)
        or isinstance(robustness_seed, bool)
        or robustness_seed < 0
    ):
        raise ConfigurationError("'robustness.seed' must be a nonnegative integer.")
    removal_fractions = _strict_float_sequence(
        robustness_raw.get("removal_fractions", [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
        key="robustness.removal_fractions",
        minimum=0.0,
        maximum=1.0,
        include_minimum=True,
    )
    if removal_fractions[0] != 0.0:
        raise ConfigurationError("'robustness.removal_fractions' must begin with 0.0.")
    robustness = RobustnessConfig(
        enabled=robustness_enabled,
        ranking_criterion=ranking_criterion,
        removal_fractions=removal_fractions,
        random_trials=random_trials,
        seed=robustness_seed,
        plots_enabled=robustness_plots,
    )

    if (density_sensitivity.enabled or robustness.enabled) and not graph.enabled:
        raise ConfigurationError(
            "'graph.enabled' must be true when sensitivity or robustness analysis is enabled."
        )

    reporting_raw = _mapping(root.get("reporting"), "reporting")
    _known_keys(reporting_raw, {"enabled"}, "reporting")
    reporting_enabled = reporting_raw.get("enabled", True)
    if not isinstance(reporting_enabled, bool):
        raise ConfigurationError("'reporting.enabled' must be true or false.")
    reporting = ReportingConfig(enabled=reporting_enabled)

    output_raw = _mapping(root.get("outputs"), "outputs")
    _known_keys(output_raw, {"directory"}, "outputs")
    output_value = output_raw.get("directory", "outputs")
    output_path = _resolve_path(output_value, base, "outputs.directory")
    seed = root.get("random_seed", 1729)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigurationError("'random_seed' must be a nonnegative integer.")

    return PipelineConfig(
        input=InputConfig(
            source_type=source_type,
            path=input_path,
            matrix_semantics=semantics,
            labels_path=labels,
            embedded_labels=embedded,
            atlas=atlas,
            symmetry_tolerance=float(symmetry_tolerance),
            correlation_tolerance=float(correlation_tolerance),
        ),
        connectivity=connectivity,
        graph=graph,
        topology=topology,
        community=community,
        smallworld=smallworld,
        density_sensitivity=density_sensitivity,
        robustness=robustness,
        outputs=OutputConfig(output_path),
        random_seed=seed,
        reporting=reporting,
    )

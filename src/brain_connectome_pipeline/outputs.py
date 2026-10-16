"""Unified-run output layout, strict JSON, and reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brain_connectome_pipeline import __version__
from brain_connectome_pipeline.config import PipelineConfig
from brain_connectome_pipeline.models import ConnectomeData


class OutputLayoutError(OSError):
    """Raised when unified-run artifacts cannot be prepared or written safely."""


@dataclass(frozen=True)
class RunOutputLayout:
    """Deterministic, input-scoped paths for a complete pipeline execution."""

    root: Path
    graph: Path
    topology: Path
    communities: Path
    smallworld: Path
    sensitivity: Path
    robustness: Path
    figures: Path
    effective_config: Path
    provenance: Path
    report: Path
    manifest: Path

    @classmethod
    def create(cls, root: str | Path) -> RunOutputLayout:
        resolved = Path(root).resolve()
        return cls(
            root=resolved,
            graph=resolved / "graph",
            topology=resolved / "topology",
            communities=resolved / "communities",
            smallworld=resolved / "smallworld",
            sensitivity=resolved / "sensitivity",
            robustness=resolved / "robustness",
            figures=resolved / "figures",
            effective_config=resolved / "effective_config.json",
            provenance=resolved / "provenance.json",
            report=resolved / "report.html",
            manifest=resolved / "execution_manifest.json",
        )


def input_stem(path: Path) -> str:
    """Return a portable, deterministic output-directory stem for one input file."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem or "connectome"


def default_output_root(config: PipelineConfig) -> Path:
    """Place non-overridden unified runs under the configured output directory."""
    return config.outputs.directory / input_stem(config.input.path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def effective_config_dict(config: PipelineConfig) -> dict[str, Any]:
    """Return validated settings while omitting machine-specific absolute path prefixes."""
    content = _json_value(asdict(config))
    input_section = content["input"]
    output_section = content["outputs"]
    assert isinstance(input_section, dict)
    assert isinstance(output_section, dict)
    input_section["path"] = config.input.path.name
    if config.input.labels_path is not None:
        input_section["labels_path"] = config.input.labels_path.name
    output_section["directory"] = config.outputs.directory.name
    content["path_representation"] = "filenames without machine-specific absolute prefixes"
    return content


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_versions() -> dict[str, str | None]:
    """Collect only the runtime versions used to reproduce a local run."""
    return {
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "networkx": _version("networkx"),
        "matplotlib": _version("matplotlib"),
        "PyYAML": _version("PyYAML"),
    }


def write_json(path: str | Path, content: dict[str, Any]) -> Path:
    """Write strict, portable JSON without nonfinite numeric literals."""
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        raise OutputLayoutError(f"Could not write JSON output '{destination}': {exc}") from exc
    return destination


def prepare_output_layout(
    root: str | Path,
    *,
    overwrite: bool,
    root_artifacts: tuple[str, ...] = (
        "effective_config.json",
        "provenance.json",
        "report.html",
        "execution_manifest.json",
    ),
) -> RunOutputLayout:
    """Preflight root-level collisions before costly analysis is started.

    Existing unrelated files are preserved.  An existing pipeline completion
    manifest, or a collision with a root artifact, requires explicit overwrite.
    Individual stage writers retain their own all-file collision checks.
    """
    layout = RunOutputLayout.create(root)
    collisions = [layout.root / name for name in root_artifacts if (layout.root / name).exists()]
    if layout.manifest.exists() and layout.manifest not in collisions:
        collisions.append(layout.manifest)
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise OutputLayoutError(f"Refusing to overwrite existing pipeline output(s): {names}")
    try:
        layout.root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputLayoutError(
            f"Could not create output directory '{layout.root}': {exc}"
        ) from exc
    return layout


def preflight_stage_collisions(
    layout: RunOutputLayout, config: PipelineConfig, *, overwrite: bool
) -> None:
    """Detect known stage-file collisions before graph calculations begin."""
    expected = [
        layout.graph / "edges.tsv",
        layout.graph / "nodes.tsv",
        layout.graph / "graph_diagnostics.json",
        layout.figures / "connectivity_matrix.png",
        layout.figures / "thresholded_binary_adjacency.png",
    ]
    if config.graph.write_binary:
        expected.append(layout.graph / "binary_adjacency.tsv")
    if config.graph.write_weighted:
        expected.append(layout.graph / "weighted_adjacency.tsv")
    if config.topology.enabled:
        expected.extend(
            [
                layout.topology / "global_metrics.tsv",
                layout.topology / "nodal_metrics.tsv",
                layout.topology / "topology_results.json",
                layout.figures / "hub_strength.png",
            ]
        )
    if config.community.enabled:
        expected.extend(
            [
                layout.communities / "community_assignments.tsv",
                layout.communities / "community_summary.json",
                layout.figures / "community_network.png",
            ]
        )
        if config.community.participation_enabled:
            expected.append(layout.communities / "participation_metrics.tsv")
        if config.community.hub_enabled:
            expected.extend(
                [layout.communities / "hub_candidates.tsv", layout.communities / "hub_summary.json"]
            )
    if config.smallworld.enabled:
        expected.extend(
            [
                layout.smallworld / "smallworld_summary.json",
                layout.smallworld / "smallworld_nulls.tsv",
            ]
        )
    if config.density_sensitivity.enabled:
        expected.extend(
            [
                layout.sensitivity / "density_sensitivity.tsv",
                layout.sensitivity / "density_sensitivity_summary.json",
            ]
        )
        if config.density_sensitivity.plots_enabled:
            expected.extend(
                [
                    layout.sensitivity / "density_global_efficiency.png",
                    layout.sensitivity / "density_clustering.png",
                ]
            )
    if config.robustness.enabled:
        expected.extend(
            [
                layout.robustness / "robustness_summary.json",
                layout.robustness / "robustness_trials.tsv",
                layout.robustness / "robustness_summary.tsv",
                layout.robustness / "robustness_ranking.tsv",
            ]
        )
        if config.robustness.plots_enabled:
            expected.extend(
                [
                    layout.robustness / "robustness_largest_component.png",
                    layout.robustness / "robustness_fixed_efficiency.png",
                ]
            )
    collisions = [path for path in expected if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(str(path.relative_to(layout.root)) for path in collisions)
        raise OutputLayoutError(f"Refusing to overwrite existing pipeline output(s): {names}")


def build_provenance(
    config: PipelineConfig,
    connectome: ConnectomeData,
    *,
    stage_status: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Create machine-readable run provenance without exposing absolute input paths."""
    return {
        "pipeline_version": __version__,
        "completion_status": "in_progress",
        "input": {
            "filename": connectome.source_path.name,
            "source_type": connectome.input_type,
            "matrix_semantics": connectome.matrix_semantics,
            "matrix_dimensions": [connectome.number_of_rois, connectome.number_of_rois],
            "roi_identifier_source": connectome.label_source,
            "roi_identifiers": list(connectome.roi_ids),
            "atlas": connectome.atlas,
            "coordinates_available": any(value is not None for value in connectome.coordinates),
            "number_of_observations": connectome.number_of_observations,
            "estimator": connectome.estimator_metadata,
        },
        "graph_configuration": {
            "mode": config.graph.mode,
            "target_density": config.graph.target_density,
            "positive_edge_policy": config.graph.positive_edge_policy,
            "tie_handling": config.graph.tie_handling,
        },
        "random_seeds": {
            "pipeline": config.random_seed,
            "community": config.community.seed,
            "smallworld": config.smallworld.seed,
            "robustness": config.robustness.seed,
        },
        "stage_status": stage_status,
        "warnings": warnings,
        "software_versions": software_versions(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
    }


def relative_outputs(layout: RunOutputLayout, paths: dict[str, Path]) -> dict[str, str]:
    """Return deterministic paths relative to the run root."""
    return {
        name: path.resolve().relative_to(layout.root).as_posix() for name, path in paths.items()
    }

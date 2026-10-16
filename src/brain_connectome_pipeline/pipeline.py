"""Unified execution of the validated connectome analysis stages."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brain_connectome_pipeline.communities import (
    CommunityResults,
    compute_community_node_metrics,
    detect_communities,
    write_community_outputs,
)
from brain_connectome_pipeline.config import PipelineConfig
from brain_connectome_pipeline.connectivity import load_connectome
from brain_connectome_pipeline.graph import construct_graphs, write_graph_outputs
from brain_connectome_pipeline.hubs import HubResults, characterize_hubs
from brain_connectome_pipeline.metrics import (
    TopologyResults,
    analyze_topology,
    write_topology_outputs,
)
from brain_connectome_pipeline.outputs import (
    OutputLayoutError,
    RunOutputLayout,
    build_provenance,
    default_output_root,
    effective_config_dict,
    preflight_stage_collisions,
    prepare_output_layout,
    relative_outputs,
    write_json,
)
from brain_connectome_pipeline.plots import write_primary_figures
from brain_connectome_pipeline.reporting import write_html_report
from brain_connectome_pipeline.robustness import (
    RobustnessResults,
    analyze_robustness,
    write_robustness_outputs,
)
from brain_connectome_pipeline.sensitivity import (
    SensitivityResults,
    analyze_density_sensitivity,
    write_sensitivity_outputs,
)
from brain_connectome_pipeline.smallworld import (
    SmallWorldResults,
    analyze_smallworld,
    write_smallworld_outputs,
)


class PipelineExecutionError(RuntimeError):
    """Raised when a unified run cannot complete a required stage."""


@dataclass(frozen=True)
class PipelineRunResults:
    """Compact outcome of one unified run, including output-relative artifacts."""

    output_root: Path
    stage_status: dict[str, dict[str, Any]]
    output_files: dict[str, str]
    warnings: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "status": "complete",
            "output_directory": str(self.output_root),
            "stage_status": self.stage_status,
            "output_files": self.output_files,
            "warnings": list(self.warnings),
        }


def _add_paths(target: dict[str, Path], prefix: str, paths: dict[str, Path]) -> None:
    target.update({f"{prefix}.{name}": path for name, path in paths.items()})


def _append_warnings(
    target: list[str], values: tuple[str, ...] | list[str], *, prefix: str | None = None
) -> None:
    for value in values:
        message = f"{prefix}: {value}" if prefix else value
        if message not in target:
            target.append(message)


def _topology_report_summary(results: TopologyResults) -> dict[str, Any]:
    strengths = [
        row
        for row in results.nodal_metrics
        if row.metric == "strength" and row.representation == "weighted" and row.value is not None
    ]
    strengths.sort(key=lambda row: (-float(row.value), row.roi_index))
    return {
        "global_metrics": [row.to_dict() for row in results.global_metrics],
        "top_strength": [
            {"roi_id": row.roi_id, "value": row.value}
            for row in strengths[: min(10, len(strengths))]
        ],
    }


def _cleanup_empty_directories(layout: RunOutputLayout) -> None:
    for directory in (
        layout.figures,
        layout.robustness,
        layout.sensitivity,
        layout.smallworld,
        layout.communities,
        layout.topology,
        layout.graph,
    ):
        with suppress(OSError):
            directory.rmdir()


def run_pipeline(
    config: PipelineConfig,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> PipelineRunResults:
    """Run configured analyses, write artifacts, and create the manifest last.

    Required input, graph, topology (when enabled), and file-writing failures
    stop the run. Optional analytical stages retain their own unavailable state
    instead of being treated as successful numerical results.
    """
    if not config.graph.enabled:
        raise PipelineExecutionError("Unified execution requires 'graph.enabled: true'.")
    connectome = load_connectome(config)
    root = Path(output_dir).resolve() if output_dir is not None else default_output_root(config)
    try:
        layout = prepare_output_layout(root, overwrite=overwrite)
        preflight_stage_collisions(layout, config, overwrite=overwrite)
    except OutputLayoutError as exc:
        raise PipelineExecutionError(str(exc)) from exc

    paths: dict[str, Path] = {}
    warnings = list(connectome.warnings)
    stage_status: dict[str, dict[str, Any]] = {}
    topology: TopologyResults | None = None
    communities: CommunityResults | None = None
    hubs: HubResults | None = None
    smallworld: SmallWorldResults | None = None
    sensitivity: SensitivityResults | None = None
    robustness: RobustnessResults | None = None

    try:
        paths["effective_config"] = write_json(
            layout.effective_config, effective_config_dict(config)
        )
        graphs = construct_graphs(connectome, config.graph)
        _append_warnings(warnings, graphs.diagnostics.warnings)
        _add_paths(
            paths,
            "graph",
            write_graph_outputs(
                graphs,
                connectome,
                layout.graph,
                write_binary=config.graph.write_binary,
                write_weighted=config.graph.write_weighted,
                overwrite=overwrite,
            ),
        )
        stage_status["graph"] = {
            "status": "complete",
            "reason": None,
            "achieved_density": graphs.diagnostics.achieved_density,
        }

        if config.topology.enabled:
            topology = analyze_topology(graphs, connectome, config.topology, config.graph)
            _append_warnings(warnings, topology.warnings, prefix="Topology")
            _add_paths(
                paths,
                "topology",
                write_topology_outputs(topology, layout.topology, overwrite=overwrite),
            )
            stage_status["topology"] = {"status": "complete", "reason": None}
        else:
            stage_status["topology"] = {"status": "skipped", "reason": "Disabled by configuration."}

        if config.community.enabled:
            communities = detect_communities(graphs, connectome, config.community)
            _append_warnings(warnings, communities.warnings, prefix="Communities")
            community_metrics = (
                compute_community_node_metrics(graphs, connectome, communities)
                if config.community.participation_enabled
                else None
            )
            if config.community.hub_enabled:
                if topology is None or community_metrics is None:
                    raise PipelineExecutionError(
                        "Candidate hub characterization requires enabled topology and "
                        "participation analysis."
                    )
                hubs = characterize_hubs(topology, community_metrics, connectome, config.community)
                if hubs.reason is not None:
                    _append_warnings(warnings, [f"Hub characterization: {hubs.reason}"])
            _add_paths(
                paths,
                "communities",
                write_community_outputs(
                    communities,
                    layout.communities,
                    node_metrics=community_metrics,
                    hub_results=hubs,
                    overwrite=overwrite,
                ),
            )
            stage_status["communities"] = {
                "status": "complete",
                "reason": None,
                "number_of_communities": communities.metadata["number_of_communities"],
            }
        else:
            stage_status["communities"] = {
                "status": "skipped",
                "reason": "Disabled by configuration.",
            }

        if config.smallworld.enabled:
            smallworld = analyze_smallworld(graphs, config.smallworld)
            _append_warnings(warnings, smallworld.warnings, prefix="Small-worldness")
            _add_paths(
                paths,
                "smallworld",
                write_smallworld_outputs(smallworld, layout.smallworld, overwrite=overwrite),
            )
            stage_status["smallworld"] = {
                "status": smallworld.status,
                "reason": smallworld.reason,
            }
        else:
            stage_status["smallworld"] = {
                "status": "skipped",
                "reason": "Disabled by configuration.",
            }

        if config.density_sensitivity.enabled:
            sensitivity = analyze_density_sensitivity(
                connectome, config.graph, config.density_sensitivity
            )
            _append_warnings(warnings, sensitivity.warnings, prefix="Density sensitivity")
            _add_paths(
                paths,
                "sensitivity",
                write_sensitivity_outputs(
                    sensitivity,
                    layout.sensitivity,
                    plots_enabled=config.density_sensitivity.plots_enabled,
                    overwrite=overwrite,
                ),
            )
            stage_status["sensitivity"] = {"status": "complete", "reason": None}
        else:
            stage_status["sensitivity"] = {
                "status": "skipped",
                "reason": "Disabled by configuration.",
            }

        if config.robustness.enabled:
            robustness = analyze_robustness(graphs, connectome, config.robustness)
            _append_warnings(warnings, robustness.warnings, prefix="Robustness")
            _add_paths(
                paths,
                "robustness",
                write_robustness_outputs(
                    robustness,
                    layout.robustness,
                    plots_enabled=config.robustness.plots_enabled,
                    overwrite=overwrite,
                ),
            )
            stage_status["robustness"] = {
                "status": robustness.status,
                "reason": robustness.reason,
            }
        else:
            stage_status["robustness"] = {
                "status": "skipped",
                "reason": "Disabled by configuration.",
            }

        _add_paths(
            paths,
            "figures",
            write_primary_figures(
                connectome,
                graphs,
                layout.figures,
                topology=topology,
                communities=communities,
                hubs=hubs,
                layout_seed=config.random_seed,
                overwrite=overwrite,
            ),
        )
        stage_status["figures"] = {"status": "complete", "reason": None}

        provenance = build_provenance(
            config,
            connectome,
            stage_status=stage_status,
            warnings=warnings,
        )
        paths["provenance"] = write_json(layout.provenance, provenance)
        report_input = {
            "filename": connectome.source_path.name,
            "input_type": connectome.input_type,
            "matrix_semantics": connectome.matrix_semantics,
            "number_of_rois": connectome.number_of_rois,
            "number_of_observations": connectome.number_of_observations,
            "label_source": connectome.label_source,
            "atlas": connectome.atlas,
            "coordinates_available": any(value is not None for value in connectome.coordinates),
            "estimator": connectome.estimator_metadata.get("estimator"),
        }
        graph_summary = graphs.diagnostics.to_dict() | {
            "positive_edge_policy": config.graph.positive_edge_policy
        }
        current_files = relative_outputs(layout, paths)
        current_files["report"] = layout.report.relative_to(layout.root).as_posix()
        current_files["manifest"] = layout.manifest.relative_to(layout.root).as_posix()
        if config.reporting.enabled:
            report_stage_status = stage_status | {"report": {"status": "complete", "reason": None}}
            paths["report"] = write_html_report(
                layout.report,
                input_summary=report_input,
                graph_summary=graph_summary,
                topology_summary=_topology_report_summary(topology)
                if topology is not None
                else None,
                community_summary=communities.summary_dict() if communities is not None else None,
                hub_summary=hubs.summary_dict() if hubs is not None else None,
                smallworld_summary=smallworld.summary_dict() if smallworld is not None else None,
                sensitivity_summary=sensitivity.summary_dict() if sensitivity is not None else None,
                robustness_summary=robustness.summary_dict() if robustness is not None else None,
                stage_status=report_stage_status,
                provenance=provenance,
                figures={
                    name: path
                    for name, path in paths.items()
                    if name.startswith("figures.") or name.endswith("_plot")
                },
                output_files=current_files,
            )
            stage_status["report"] = {"status": "complete", "reason": None}
        else:
            stage_status["report"] = {
                "status": "skipped",
                "reason": "Disabled by configuration.",
            }

        provenance["completion_status"] = "complete"
        provenance["stage_status"] = stage_status
        paths["provenance"] = write_json(layout.provenance, provenance)
        output_files = relative_outputs(layout, paths)
        output_files["manifest"] = layout.manifest.relative_to(layout.root).as_posix()
        manifest = {
            "completion_status": "complete",
            "pipeline_version": provenance["pipeline_version"],
            "input_filename": connectome.source_path.name,
            "stage_status": stage_status,
            "output_files": output_files,
            "warnings": warnings,
        }
        paths["manifest"] = write_json(layout.manifest, manifest)
        return PipelineRunResults(layout.root, stage_status, output_files, tuple(warnings))
    except PipelineExecutionError:
        _cleanup_empty_directories(layout)
        raise
    except Exception as exc:
        _cleanup_empty_directories(layout)
        raise PipelineExecutionError(f"Unified pipeline failed: {exc}") from exc

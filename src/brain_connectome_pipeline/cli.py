"""Command-line interface for connectome graph analyses."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from brain_connectome_pipeline import __version__
from brain_connectome_pipeline.communities import (
    CommunityAnalysisError,
    CommunityOutputError,
    compute_community_node_metrics,
    detect_communities,
    write_community_outputs,
)
from brain_connectome_pipeline.config import ConfigurationError, load_config
from brain_connectome_pipeline.connectivity import load_connectome
from brain_connectome_pipeline.graph import (
    GraphConstructionError,
    GraphOutputError,
    construct_graphs,
    write_graph_outputs,
)
from brain_connectome_pipeline.hubs import HubAnalysisError, characterize_hubs
from brain_connectome_pipeline.io import InputValidationError
from brain_connectome_pipeline.metrics import (
    TopologyAnalysisError,
    TopologyOutputError,
    analyze_topology,
    write_topology_outputs,
)
from brain_connectome_pipeline.pipeline import PipelineExecutionError, run_pipeline
from brain_connectome_pipeline.robustness import (
    RobustnessAnalysisError,
    RobustnessOutputError,
    analyze_robustness,
    write_robustness_outputs,
)
from brain_connectome_pipeline.sensitivity import (
    SensitivityAnalysisError,
    SensitivityOutputError,
    analyze_density_sensitivity,
    write_sensitivity_outputs,
)
from brain_connectome_pipeline.smallworld import (
    SmallWorldAnalysisError,
    SmallWorldOutputError,
    analyze_smallworld,
    write_smallworld_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain-connectome-pipeline",
        description="Validate connectomes and analyze graph topology and organization.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("version", help="Show package version.")
    validate = commands.add_parser("validate-config", help="Validate a YAML configuration.")
    validate.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect-input", help="Validate and summarize the configured input."
    )
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    build = commands.add_parser(
        "build-graph", help="Construct and write configured binary and weighted graphs."
    )
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, help="Override the configured output directory.")
    build.add_argument("--overwrite", action="store_true", help="Replace existing graph outputs.")
    build.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    topology = commands.add_parser(
        "analyze-topology", help="Compute and write global and nodal topology metrics."
    )
    topology.add_argument("--config", type=Path, required=True)
    topology.add_argument(
        "--output-dir", type=Path, help="Override the configured output directory."
    )
    topology.add_argument(
        "--overwrite", action="store_true", help="Replace existing topology outputs."
    )
    topology.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    communities = commands.add_parser(
        "analyze-communities",
        help="Detect weighted communities and characterize participation and candidate hubs.",
    )
    communities.add_argument("--config", type=Path, required=True)
    communities.add_argument(
        "--output-dir", type=Path, help="Override the configured output directory."
    )
    communities.add_argument(
        "--overwrite", action="store_true", help="Replace existing community outputs."
    )
    communities.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    smallworld = commands.add_parser(
        "analyze-smallworld",
        help="Estimate classical binary small-worldness from degree-matched null graphs.",
    )
    smallworld.add_argument("--config", type=Path, required=True)
    smallworld.add_argument(
        "--output-dir", type=Path, help="Override the configured output directory."
    )
    smallworld.add_argument(
        "--overwrite", action="store_true", help="Replace existing small-world outputs."
    )
    smallworld.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    sensitivity = commands.add_parser(
        "analyze-sensitivity",
        help="Evaluate predefined topology metrics across proportional graph densities.",
    )
    sensitivity.add_argument("--config", type=Path, required=True)
    sensitivity.add_argument(
        "--output-dir", type=Path, help="Override the configured output directory."
    )
    sensitivity.add_argument(
        "--overwrite", action="store_true", help="Replace existing sensitivity outputs."
    )
    sensitivity.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    robustness = commands.add_parser(
        "analyze-robustness",
        help="Compare fixed targeted and repeated random cumulative node removal.",
    )
    robustness.add_argument("--config", type=Path, required=True)
    robustness.add_argument(
        "--output-dir", type=Path, help="Override the configured output directory."
    )
    robustness.add_argument(
        "--overwrite", action="store_true", help="Replace existing robustness outputs."
    )
    robustness.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    run = commands.add_parser(
        "run",
        help="Run configured graph analyses and write a unified offline report.",
    )
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, help="Override the input-scoped output directory.")
    run.add_argument("--overwrite", action="store_true", help="Replace colliding pipeline outputs.")
    run.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    return parser


def _human_inspection(data: dict[str, object]) -> str:
    diagnostics = data["diagnostics"]
    assert isinstance(diagnostics, dict)
    lines = [
        f"Input source type: {data['input_type']}",
        f"Path: {data['source_path']}",
        f"Number of ROIs: {data['number_of_rois']}",
        f"Matrix semantics: {data['matrix_semantics']}",
        f"Symmetry: {'valid' if diagnostics['symmetric'] else 'invalid'} "
        f"(maximum asymmetry {diagnostics['maximum_asymmetry']:.6g})",
        f"Diagonal status: {diagnostics['diagonal_status']}",
        f"Positive edges: {diagnostics['positive_edges']}",
        f"Negative edges: {diagnostics['negative_edges']}",
        f"Zero edges: {diagnostics['zero_edges']}",
        f"ROI label source: {data['label_source']}",
    ]
    if data["atlas"]:
        lines.append(f"Atlas/parcellation: {data['atlas']}")
    if data["number_of_observations"] is not None:
        lines.append(f"Number of observations: {data['number_of_observations']}")
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_graph_summary(data: dict[str, object]) -> str:
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    lines = [
        f"Graph mode: {data['threshold_mode']}",
        f"Nodes: {data['number_of_nodes']}",
        f"Eligible positive edges: {data['positive_eligible_edges']}",
        f"Retained edges: {data['retained_edges']}",
        f"Requested density: {data['requested_density']}",
        f"Achieved density: {data['achieved_density']:.6g}",
        f"Connected components: {data['connected_components']}",
        f"Isolated nodes: {data['isolated_nodes']}",
        f"Output directory: {data['output_directory']}",
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_topology_summary(data: dict[str, object]) -> str:
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    lines = [
        f"Global metrics: {data['global_metrics']}",
        f"Global metrics unavailable: {data['global_metrics_unavailable']}",
        f"Nodal metric rows: {data['nodal_metric_rows']}",
        f"Nodal metrics unavailable: {data['nodal_metrics_unavailable']}",
        f"Output directory: {data['output_directory']}",
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_community_summary(data: dict[str, object]) -> str:
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    lines = [
        f"Communities: {data['number_of_communities']}",
        f"Weighted modularity: {data['modularity']}",
        f"Modularity status: {data['modularity_status']}",
        f"Candidate hubs: {data.get('candidate_hub_count', 'not requested')}",
        f"Output directory: {data['output_directory']}",
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_smallworld_summary(data: dict[str, object]) -> str:
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    metadata = data["metadata"]
    assert isinstance(metadata, dict)
    lines = [
        f"Small-worldness status: {data['status']}",
        f"Sigma: {data['sigma']}",
        f"Accepted null graphs: {metadata['accepted_samples']}",
        f"Reason: {data['reason'] or 'none'}",
        f"Output directory: {data['output_directory']}",
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_sensitivity_summary(data: dict[str, object]) -> str:
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    lines = [
        f"Density points: {data['density_points']}",
        f"Unique graph topologies: {data['unique_topologies']}",
        f"Metric rows: {data['metric_rows']}",
        f"Output directory: {data['output_directory']}",
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_robustness_summary(data: dict[str, object]) -> str:
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    metadata = data["metadata"]
    assert isinstance(metadata, dict)
    checkpoints = data["checkpoints"]
    assert isinstance(checkpoints, list)
    lines = [
        f"Robustness status: {data['status']}",
        f"Removal checkpoints: {len(checkpoints)}",
        f"Random trials: {metadata['random_trials']}",
        f"Reason: {data['reason'] or 'none'}",
        f"Output directory: {data['output_directory']}",
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def _human_pipeline_summary(data: dict[str, object]) -> str:
    stages = data["stage_status"]
    assert isinstance(stages, dict)
    lines = [
        "Unified analysis complete.",
        f"Output directory: {data['output_directory']}",
        "Stages:",
    ]
    for stage, details in stages.items():
        assert isinstance(details, dict)
        reason = details.get("reason")
        lines.append(f"  {stage}: {details.get('status')}{f' ({reason})' if reason else ''}")
    warnings = data["warnings"]
    assert isinstance(warnings, list)
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "version":
        print(__version__)
        return 0
    try:
        config = load_config(args.config)
        if args.command == "validate-config":
            print(f"Configuration valid: {Path(args.config).resolve()}")
            return 0
        if args.command == "run":
            results = run_pipeline(
                config,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            summary = results.summary_dict()
            print(json.dumps(summary, indent=2) if args.json else _human_pipeline_summary(summary))
            return 0
        connectome = load_connectome(config)
        if args.command == "inspect-input":
            inspection = connectome.inspection_dict()
            print(json.dumps(inspection, indent=2) if args.json else _human_inspection(inspection))
            return 0
        graphs = construct_graphs(connectome, config.graph)
        output_dir = args.output_dir if args.output_dir is not None else config.outputs.directory
        if args.command == "build-graph":
            paths = write_graph_outputs(
                graphs,
                connectome,
                output_dir,
                write_binary=config.graph.write_binary,
                write_weighted=config.graph.write_weighted,
                overwrite=args.overwrite,
            )
            summary = graphs.summary_dict()
            summary["output_directory"] = str(Path(output_dir).resolve())
            summary["output_files"] = {name: str(path) for name, path in paths.items()}
            print(json.dumps(summary, indent=2) if args.json else _human_graph_summary(summary))
            return 0
        if args.command == "analyze-topology":
            topology_results = analyze_topology(graphs, connectome, config.topology, config.graph)
            paths = write_topology_outputs(topology_results, output_dir, overwrite=args.overwrite)
            summary = topology_results.summary_dict()
            human_summary = _human_topology_summary
        elif args.command == "analyze-communities":
            community_results = detect_communities(graphs, connectome, config.community)
            node_metrics = (
                compute_community_node_metrics(graphs, connectome, community_results)
                if config.community.participation_enabled
                else None
            )
            hub_results = None
            if config.community.hub_enabled:
                if node_metrics is None:
                    raise CommunityAnalysisError(
                        "Hub characterization requires participation metrics."
                    )
                topology_results = analyze_topology(
                    graphs, connectome, config.topology, config.graph
                )
                hub_results = characterize_hubs(
                    topology_results, node_metrics, connectome, config.community
                )
            paths = write_community_outputs(
                community_results,
                output_dir,
                node_metrics=node_metrics,
                hub_results=hub_results,
                overwrite=args.overwrite,
            )
            summary = community_results.summary_dict()
            if hub_results is not None:
                summary["hub_status"] = hub_results.status
                summary["candidate_hub_count"] = sum(
                    row.is_candidate_hub for row in hub_results.rows
                )
            human_summary = _human_community_summary
        elif args.command == "analyze-smallworld":
            smallworld_results = analyze_smallworld(graphs, config.smallworld)
            paths = write_smallworld_outputs(
                smallworld_results, output_dir, overwrite=args.overwrite
            )
            summary = smallworld_results.summary_dict()
            human_summary = _human_smallworld_summary
        elif args.command == "analyze-sensitivity":
            sensitivity_results = analyze_density_sensitivity(
                connectome, config.graph, config.density_sensitivity
            )
            paths = write_sensitivity_outputs(
                sensitivity_results,
                output_dir,
                plots_enabled=config.density_sensitivity.plots_enabled,
                overwrite=args.overwrite,
            )
            summary = sensitivity_results.summary_dict()
            human_summary = _human_sensitivity_summary
        elif args.command == "analyze-robustness":
            robustness_results = analyze_robustness(graphs, connectome, config.robustness)
            paths = write_robustness_outputs(
                robustness_results,
                output_dir,
                plots_enabled=config.robustness.plots_enabled,
                overwrite=args.overwrite,
            )
            summary = robustness_results.summary_dict()
            human_summary = _human_robustness_summary
        else:
            raise RuntimeError(f"Unhandled command: {args.command}")
        summary["output_directory"] = str(Path(output_dir).resolve())
        summary["output_files"] = {name: str(path) for name, path in paths.items()}
        print(json.dumps(summary, indent=2) if args.json else human_summary(summary))
        return 0
    except (
        ConfigurationError,
        InputValidationError,
        GraphConstructionError,
        GraphOutputError,
        TopologyAnalysisError,
        TopologyOutputError,
        CommunityAnalysisError,
        CommunityOutputError,
        HubAnalysisError,
        SmallWorldAnalysisError,
        SmallWorldOutputError,
        SensitivityAnalysisError,
        SensitivityOutputError,
        RobustnessAnalysisError,
        RobustnessOutputError,
        PipelineExecutionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

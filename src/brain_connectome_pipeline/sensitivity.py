"""Independent proportional-density graph sensitivity analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from brain_connectome_pipeline.config import (
    DensitySensitivityConfig,
    GraphConfig,
    TopologyConfig,
)
from brain_connectome_pipeline.graph import construct_graphs
from brain_connectome_pipeline.metrics import GlobalMetricResult, compute_global_metrics
from brain_connectome_pipeline.models import ConnectomeData


class SensitivityAnalysisError(ValueError):
    """Raised when density sensitivity cannot be calculated safely."""


class SensitivityOutputError(OSError):
    """Raised when sensitivity outputs cannot be written safely."""


@dataclass(frozen=True)
class DensityMetricRow:
    density_index: int
    requested_density: float
    achieved_density: float
    retained_edges: int
    topology_id: str
    repeated_topology: bool
    metric: str
    representation: Literal["binary", "weighted"]
    value: float | int | None
    status: Literal["available", "unavailable"]
    reason: str | None
    scope: str
    graph_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "density_index": self.density_index,
            "requested_density": self.requested_density,
            "achieved_density": self.achieved_density,
            "retained_edges": self.retained_edges,
            "topology_id": self.topology_id,
            "repeated_topology": self.repeated_topology,
            "metric": self.metric,
            "representation": self.representation,
            "value": self.value,
            "status": self.status,
            "reason": self.reason,
            "scope": self.scope,
            "graph_warnings": " | ".join(self.graph_warnings),
        }


@dataclass(frozen=True)
class SensitivityResults:
    rows: tuple[DensityMetricRow, ...]
    density_records: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]

    def summary_dict(self) -> dict[str, Any]:
        ranges: dict[str, dict[str, float]] = {}
        keys = sorted({(row.metric, row.representation) for row in self.rows})
        for metric, representation in keys:
            values = [
                float(row.value)
                for row in self.rows
                if row.metric == metric
                and row.representation == representation
                and row.status == "available"
                and row.value is not None
            ]
            if values:
                ranges[f"{representation}.{metric}"] = {
                    "minimum": min(values),
                    "maximum": max(values),
                    "last_minus_first": values[-1] - values[0],
                }
        groups: dict[str, list[float]] = {}
        for record in self.density_records:
            groups.setdefault(str(record["topology_id"]), []).append(
                float(record["requested_density"])
            )
        repeated = [
            {"topology_id": topology_id, "requested_densities": densities}
            for topology_id, densities in groups.items()
            if len(densities) > 1
        ]
        return {
            "status": "available",
            "density_points": len(self.density_records),
            "metric_rows": len(self.rows),
            "unique_topologies": len(groups),
            "repeated_topologies": repeated,
            "metric_ranges": ranges,
            "density_records": list(self.density_records),
            "metadata": self.metadata,
            "warnings": list(self.warnings),
        }


_METRICS = (
    ("global_efficiency", "binary"),
    ("global_efficiency", "weighted"),
    ("mean_clustering_coefficient", "binary"),
    ("mean_clustering_coefficient", "weighted"),
    ("characteristic_path_length", "binary"),
    ("largest_component_fraction", "binary"),
    ("connected_components", "binary"),
    ("mean_node_strength", "weighted"),
)


def _selected_metric(
    metrics: tuple[GlobalMetricResult, ...], metric: str, representation: str
) -> GlobalMetricResult:
    matches = [
        row
        for row in metrics
        if row.metric == metric
        and row.representation == representation
        and row.scope == "whole_graph"
    ]
    if len(matches) != 1:
        raise SensitivityAnalysisError(
            f"Expected one whole-graph {representation} metric {metric!r}; found {len(matches)}."
        )
    return matches[0]


def analyze_density_sensitivity(
    connectome: ConnectomeData,
    graph_config: GraphConfig,
    config: DensitySensitivityConfig,
) -> SensitivityResults:
    """Rebuild independent proportional graphs and select predefined global metrics."""
    if not config.enabled:
        raise SensitivityAnalysisError("Density sensitivity is disabled in the configuration.")
    if not graph_config.enabled:
        raise SensitivityAnalysisError(
            "Graph construction must be enabled for sensitivity analysis."
        )
    topology_config = TopologyConfig(enabled=True, report_largest_component_paths=False)
    provisional: list[dict[str, Any]] = []
    signatures: dict[tuple[tuple[int, int], ...], str] = {}
    topology_counts: dict[str, int] = {}
    all_warnings: list[str] = []
    for density_index, requested_density in enumerate(config.densities):
        current_config = replace(
            graph_config,
            enabled=True,
            mode="proportional",
            target_density=requested_density,
        )
        graphs = construct_graphs(connectome, current_config)
        if tuple(graphs.binary_graph.nodes) != connectome.roi_ids:
            raise SensitivityAnalysisError("ROI order changed during density reconstruction.")
        signature = tuple((edge.source_index, edge.target_index) for edge in graphs.selected_edges)
        topology_id = signatures.setdefault(signature, f"topology_{len(signatures) + 1:03d}")
        topology_counts[topology_id] = topology_counts.get(topology_id, 0) + 1
        global_metrics = compute_global_metrics(graphs, topology_config, connectome.roi_ids)
        for warning in graphs.diagnostics.warnings:
            if warning not in all_warnings:
                all_warnings.append(warning)
        provisional.append(
            {
                "density_index": density_index,
                "requested_density": requested_density,
                "achieved_density": graphs.diagnostics.achieved_density,
                "retained_edges": graphs.diagnostics.retained_edges,
                "topology_id": topology_id,
                "metrics": global_metrics,
                "warnings": graphs.diagnostics.warnings,
            }
        )
    rows: list[DensityMetricRow] = []
    records: list[dict[str, Any]] = []
    for point in provisional:
        repeated = topology_counts[str(point["topology_id"])] > 1
        records.append(
            {
                "density_index": point["density_index"],
                "requested_density": point["requested_density"],
                "achieved_density": point["achieved_density"],
                "retained_edges": point["retained_edges"],
                "topology_id": point["topology_id"],
                "repeated_topology": repeated,
                "warnings": list(point["warnings"]),
            }
        )
        for metric, representation in _METRICS:
            selected = _selected_metric(point["metrics"], metric, representation)
            value = selected.value
            if isinstance(value, float) and not math.isfinite(value):
                raise SensitivityAnalysisError(
                    f"Sensitivity metric {representation}.{metric} was nonfinite."
                )
            rows.append(
                DensityMetricRow(
                    int(point["density_index"]),
                    float(point["requested_density"]),
                    float(point["achieved_density"]),
                    int(point["retained_edges"]),
                    str(point["topology_id"]),
                    repeated,
                    metric,
                    representation,  # type: ignore[arg-type]
                    value,
                    selected.status,
                    selected.reason,
                    selected.scope,
                    tuple(point["warnings"]),
                )
            )
    metadata = {
        "analysis": "graph-density sensitivity",
        "density_definition": "retained undirected edges divided by N(N-1)/2",
        "edge_count_rounding": "floor(requested_density * possible_edges + 0.5)",
        "graph_construction": "independent reconstruction from the same original matrix",
        "tie_handling": graph_config.tie_handling,
        "positive_edge_policy": graph_config.positive_edge_policy,
        "roi_order": list(connectome.roi_ids),
        "not_independent_observations": True,
    }
    return SensitivityResults(tuple(rows), tuple(records), metadata, tuple(all_warnings))


def write_sensitivity_outputs(
    results: SensitivityResults,
    output_dir: str | Path,
    *,
    plots_enabled: bool,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write sensitivity tables, strict JSON, and optional diagnostic plots."""
    directory = Path(output_dir).resolve()
    paths = {
        "density_sensitivity": directory / "density_sensitivity.tsv",
        "density_sensitivity_summary": directory / "density_sensitivity_summary.json",
    }
    if plots_enabled:
        paths["density_efficiency_plot"] = directory / "density_global_efficiency.png"
        paths["density_clustering_plot"] = directory / "density_clustering.png"
    collisions = [path for path in paths.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise SensitivityOutputError(
            f"Refusing to overwrite existing sensitivity output(s): {names}"
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row.to_dict() for row in results.rows]).to_csv(
            paths["density_sensitivity"], sep="\t", index=False, float_format="%.17g"
        )
        paths["density_sensitivity_summary"].write_text(
            json.dumps(results.summary_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if plots_enabled:
            from brain_connectome_pipeline.plots import plot_density_sensitivity

            plot_density_sensitivity(
                results,
                paths["density_efficiency_plot"],
                paths["density_clustering_plot"],
            )
    except (OSError, ValueError) as exc:
        raise SensitivityOutputError(
            f"Could not write sensitivity outputs to '{directory}': {exc}"
        ) from exc
    return paths

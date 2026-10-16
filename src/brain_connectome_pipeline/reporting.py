"""Self-contained offline HTML reporting from computed pipeline outputs."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any


class ReportingError(OSError):
    """Raised when an offline analysis report cannot be produced."""


def _display(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(_display(item) for item in value)
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return '<p class="unavailable">No available rows.</p>'
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_display(row.get(key)))}</td>" for key, _ in columns)
        + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _figure(path: Path, caption: str) -> str:
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ReportingError(f"Could not embed figure '{path}': {exc}") from exc
    return (
        '<figure><img alt="'
        + html.escape(caption)
        + '" src="data:image/png;base64,'
        + encoded
        + '"><figcaption>'
        + html.escape(caption)
        + "</figcaption></figure>"
    )


def _section(title: str, content: str) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{content}</section>"


def write_html_report(
    path: str | Path,
    *,
    input_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    topology_summary: dict[str, Any] | None,
    community_summary: dict[str, Any] | None,
    hub_summary: dict[str, Any] | None,
    smallworld_summary: dict[str, Any] | None,
    sensitivity_summary: dict[str, Any] | None,
    robustness_summary: dict[str, Any] | None,
    stage_status: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    figures: dict[str, Path],
    output_files: dict[str, str],
) -> Path:
    """Write an offline report using only calculated values and embedded PNGs."""
    destination = Path(path)
    input_rows = [
        {"item": "Input filename", "value": input_summary["filename"]},
        {"item": "Input type", "value": input_summary["input_type"]},
        {"item": "Matrix semantics", "value": input_summary["matrix_semantics"]},
        {"item": "ROIs", "value": input_summary["number_of_rois"]},
        {"item": "Observations", "value": input_summary["number_of_observations"]},
        {"item": "ROI identifier source", "value": input_summary["label_source"]},
        {"item": "Atlas/parcellation", "value": input_summary["atlas"]},
        {"item": "Coordinates supplied", "value": input_summary["coordinates_available"]},
        {"item": "Estimator", "value": input_summary["estimator"]},
    ]
    graph_rows = [
        {"item": "Positive-edge policy", "value": graph_summary["positive_edge_policy"]},
        {"item": "Threshold mode", "value": graph_summary["threshold_mode"]},
        {"item": "Requested density", "value": graph_summary["requested_density"]},
        {"item": "Achieved density", "value": graph_summary["achieved_density"]},
        {"item": "Retained edges", "value": graph_summary["retained_edges"]},
        {"item": "Excluded negative edges", "value": graph_summary["negative_excluded_edges"]},
        {"item": "Zero entries", "value": graph_summary["zero_entries"]},
        {"item": "Connected components", "value": graph_summary["connected_components"]},
        {"item": "Isolated nodes", "value": graph_summary["isolated_nodes"]},
    ]
    content = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Brain Connectome Pipeline analysis report</title><style>"
        "body{font-family:Arial,sans-serif;line-height:1.45;margin:0 auto;max-width:1120px;padding:2rem;color:#17202a}"
        "h1{margin-bottom:.2rem}h2{border-bottom:2px solid #d5dde5;padding-bottom:.25rem;margin-top:2rem}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #d5dde5;padding:.4rem;text-align:left;vertical-align:top}th{background:#eef3f7}"
        ".table-wrap{overflow-x:auto}.note{background:#fff8df;border-left:4px solid #c99500;padding:.7rem}.unavailable{color:#7a1d1d}"
        "figure{margin:1.2rem 0}figure img{max-width:100%;height:auto;border:1px solid #d5dde5}figcaption{font-size:.9rem;color:#445}"
        "code,pre{background:#f5f7f9;padding:.2rem .35rem}pre{overflow:auto;padding:.8rem}</style></head><body>"
        "<h1>Brain Connectome Pipeline</h1><p>Functional Network Topology, Community Structure &amp; Robustness Analysis</p>"
        "<p>This offline report describes graph-theoretic results for the configured functional connectome. "
        "They depend on the input, correlation estimation, ROI definition, and graph-construction choices; they are not clinical or anatomical findings.</p>"
    ]
    content.append(
        _section("Input and provenance", _table(input_rows, [("item", "Item"), ("value", "Value")]))
    )
    content.append(
        '<p class="note">Time-series inputs must already have undergone appropriate upstream processing. '
        "This pipeline does not perform motion correction, confound control, filtering, scrubbing, temporal alignment, or atlas extraction.</p>"
    )
    content.append(
        _section("Graph construction", _table(graph_rows, [("item", "Item"), ("value", "Value")]))
    )
    content.append(
        '<p class="note">Only strictly positive retained values become graph edges. Negative correlations remain excluded; '
        "their magnitudes are never converted to positive connectivity. Requested and achieved density can differ when eligible edges are insufficient.</p>"
    )
    if topology_summary is None:
        content.append(
            _section(
                "Global and nodal topology", '<p class="unavailable">Disabled by configuration.</p>'
            )
        )
    else:
        content.append(
            _section(
                "Global and nodal topology",
                _table(
                    topology_summary["global_metrics"],
                    [
                        ("metric", "Metric"),
                        ("representation", "Representation"),
                        ("value", "Value"),
                        ("status", "Status"),
                        ("scope", "Scope"),
                        ("reason", "Unavailable reason"),
                    ],
                )
                + "<p>Binary paths use unit hops. Weighted paths and weighted efficiency use reciprocal retained positive strength as distance; raw strength is not a traversal cost.</p>",
            )
        )
        nodal = topology_summary.get("top_strength", [])
        if nodal:
            content.append(
                _section(
                    "Important centrality results",
                    _table(nodal, [("roi_id", "ROI"), ("value", "Positive strength")]),
                )
            )
    if community_summary is None:
        content.append(
            _section(
                "Communities and hubs", '<p class="unavailable">Disabled by configuration.</p>'
            )
        )
    else:
        community_rows = [
            {"item": "Algorithm", "value": community_summary.get("algorithm")},
            {
                "item": "Resolution",
                "value": community_summary.get("metadata", {}).get("resolution"),
            },
            {"item": "Weighted modularity", "value": community_summary.get("modularity")},
            {"item": "Community count", "value": community_summary.get("number_of_communities")},
            {"item": "Community sizes", "value": community_summary.get("community_sizes")},
        ]
        if hub_summary is not None:
            community_rows.append(
                {"item": "Candidate hubs", "value": hub_summary.get("candidate_hubs")}
            )
            community_rows.append(
                {
                    "item": "Hub-selection rule",
                    "value": hub_summary.get("metadata", {}).get("candidate_rule"),
                }
            )
        content.append(
            _section(
                "Communities and hubs",
                _table(community_rows, [("item", "Item"), ("value", "Value")])
                + "<p>Louvain is heuristic; community and hub labels describe this constructed graph rather than verified functional systems or anatomical control centers.</p>",
            )
        )
    if smallworld_summary is None:
        content.append(
            _section("Small-worldness", '<p class="unavailable">Disabled by configuration.</p>')
        )
    else:
        smallworld_rows = [
            {"item": key.replace("_", " "), "value": value}
            for key, value in smallworld_summary.items()
            if key
            in {
                "status",
                "reason",
                "observed_clustering",
                "observed_path_length",
                "null_clustering_mean",
                "null_clustering_std",
                "null_path_length_mean",
                "null_path_length_std",
                "gamma",
                "lambda",
                "sigma",
            }
        ]
        content.append(
            _section(
                "Small-worldness",
                _table(smallworld_rows, [("item", "Item"), ("value", "Value")])
                + "<p>Classical binary sigma uses a degree-matched randomized reference ensemble. A sigma value is not proof of biological small-world organization.</p>",
            )
        )
    if sensitivity_summary is None:
        content.append(
            _section("Density sensitivity", '<p class="unavailable">Disabled by configuration.</p>')
        )
    else:
        density_rows = sensitivity_summary.get("density_records", [])
        content.append(
            _section(
                "Density sensitivity",
                _table(
                    [
                        {"item": key.replace("_", " "), "value": value}
                        for key, value in sensitivity_summary.items()
                        if key
                        in {
                            "status",
                            "density_points",
                            "unique_topologies",
                            "repeated_topologies",
                        }
                    ],
                    [("item", "Item"), ("value", "Value")],
                )
                + _table(
                    density_rows,
                    [
                        ("requested_density", "Requested density"),
                        ("achieved_density", "Achieved density"),
                        ("retained_edges", "Retained edges"),
                        ("topology_id", "Topology"),
                        ("repeats_previous_topology", "Repeated topology"),
                    ],
                )
                + "<p>Threshold points are graph constructions from one input, not independent biological observations; no inferential comparison is reported.</p>",
            )
        )
    if robustness_summary is None:
        content.append(
            _section("Network robustness", '<p class="unavailable">Disabled by configuration.</p>')
        )
    else:
        robustness_metadata = robustness_summary.get("metadata", {})
        robustness_rows = [
            row
            for row in robustness_summary.get("aggregate_results", [])
            if row.get("metric")
            in {
                "largest_component_fraction",
                "global_efficiency_original_node_normalized",
            }
        ]
        content.append(
            _section(
                "Network robustness",
                _table(
                    [
                        {"item": "Status", "value": robustness_summary.get("status")},
                        {"item": "Reason", "value": robustness_summary.get("reason")},
                        {
                            "item": "Targeted ranking",
                            "value": robustness_metadata.get("ranking_criterion"),
                        },
                        {
                            "item": "Random trials",
                            "value": robustness_metadata.get("random_trials"),
                        },
                        {
                            "item": "Checkpoint rounding",
                            "value": robustness_metadata.get("node_count_rounding"),
                        },
                    ],
                    [("item", "Item"), ("value", "Value")],
                )
                + _table(
                    robustness_rows,
                    [
                        ("strategy", "Strategy"),
                        ("removed_count", "Removed nodes"),
                        ("actual_removed_fraction", "Removed fraction"),
                        ("metric", "Metric"),
                        ("value", "Value"),
                        ("sample_standard_deviation", "Random sample SD"),
                    ],
                )
                + "<p>Targeted removal uses one fixed intact-graph ranking. Random curves summarize configured repeated node permutations. These are mathematical graph perturbations, not biological lesion models.</p>",
            )
        )
    figure_content = "".join(
        _figure(path, key.replace("_", " ").capitalize()) for key, path in figures.items()
    )
    content.append(
        _section(
            "Figures", figure_content or '<p class="unavailable">No figures were generated.</p>'
        )
    )
    status_rows = [
        {"stage": stage, "status": details.get("status"), "reason": details.get("reason")}
        for stage, details in stage_status.items()
    ]
    content.append(
        _section(
            "Reproducibility and completion",
            _table(status_rows, [("stage", "Stage"), ("status", "Status"), ("reason", "Reason")])
            + "<p>Completion manifest: <code>execution_manifest.json</code>. All listed output paths are relative to this run directory.</p>",
        )
    )
    content.append(
        _section(
            "Output files",
            _table(
                [{"artifact": key, "path": value} for key, value in output_files.items()],
                [("artifact", "Artifact"), ("path", "Relative path")],
            ),
        )
    )
    content.append(
        _section(
            "Warnings and limitations",
            "<ul>"
            + "".join(
                f"<li>{html.escape(warning)}</li>" for warning in provenance.get("warnings", [])
            )
            + "</ul>",
        )
    )
    content.append("</body></html>\n")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(content), encoding="utf-8")
    except OSError as exc:
        raise ReportingError(f"Could not write HTML report '{destination}': {exc}") from exc
    return destination

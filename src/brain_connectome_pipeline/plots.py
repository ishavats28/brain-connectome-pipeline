"""Compact diagnostic PNG plots for sensitivity and robustness analyses."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "brain_connectome_pipeline_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from brain_connectome_pipeline.communities import CommunityResults
    from brain_connectome_pipeline.graph import ConstructedGraphs
    from brain_connectome_pipeline.hubs import HubResults
    from brain_connectome_pipeline.metrics import TopologyResults
    from brain_connectome_pipeline.models import ConnectomeData
    from brain_connectome_pipeline.robustness import RobustnessResults
    from brain_connectome_pipeline.sensitivity import SensitivityResults


def _finish(figure: plt.Figure, path: str | Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _density_panel(
    results: SensitivityResults,
    metric: str,
    ylabel: str,
    title: str,
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    try:
        unavailable = 0
        for representation, marker in (("binary", "o"), ("weighted", "s")):
            rows = [
                row
                for row in results.rows
                if row.metric == metric and row.representation == representation
            ]
            available = [row for row in rows if row.status == "available" and row.value is not None]
            unavailable += len(rows) - len(available)
            if available:
                axis.plot(
                    [row.achieved_density for row in available],
                    [float(row.value) for row in available],
                    marker=marker,
                    linewidth=1.8,
                    label=representation.capitalize(),
                )
        axis.set_xlabel("Achieved graph density")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
        if unavailable:
            axis.text(
                0.01,
                0.01,
                f"Unavailable points omitted: {unavailable}",
                transform=axis.transAxes,
                fontsize=8,
                verticalalignment="bottom",
            )
        figure.text(
            0.5,
            0.005,
            "Each point is a requested threshold; x-axis positions show achieved density.",
            ha="center",
            fontsize=8,
        )
        _finish(figure, path)
    except Exception:
        plt.close(figure)
        raise


def plot_density_sensitivity(
    results: SensitivityResults,
    efficiency_path: str | Path,
    clustering_path: str | Path,
) -> None:
    """Plot binary/weighted efficiency and clustering against achieved density."""
    _density_panel(
        results,
        "global_efficiency",
        "Global efficiency",
        "Density sensitivity: global efficiency",
        efficiency_path,
    )
    _density_panel(
        results,
        "mean_clustering_coefficient",
        "Mean clustering coefficient",
        "Density sensitivity: clustering",
        clustering_path,
    )


def _robustness_panel(
    results: RobustnessResults,
    metric: str,
    ylabel: str,
    title: str,
    path: str | Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    try:
        targeted = [
            row
            for row in results.aggregate_rows
            if row.strategy == "targeted"
            and row.metric == metric
            and row.status == "available"
            and row.value is not None
        ]
        random_rows = [
            row
            for row in results.aggregate_rows
            if row.strategy == "random_mean"
            and row.metric == metric
            and row.status == "available"
            and row.value is not None
        ]
        if targeted:
            axis.plot(
                [row.actual_removed_fraction for row in targeted],
                [float(row.value) for row in targeted],
                marker="o",
                linewidth=2.0,
                label="Targeted (fixed initial ranking)",
            )
        if random_rows:
            x_values = np.asarray([row.actual_removed_fraction for row in random_rows], dtype=float)
            means = np.asarray([float(row.value) for row in random_rows], dtype=float)
            axis.plot(
                x_values,
                means,
                marker="s",
                linewidth=2.0,
                label="Random mean",
            )
            uncertainty = [row for row in random_rows if row.sample_standard_deviation is not None]
            if uncertainty:
                band_x = np.asarray(
                    [row.actual_removed_fraction for row in uncertainty], dtype=float
                )
                band_mean = np.asarray([float(row.value) for row in uncertainty], dtype=float)
                band_sd = np.asarray(
                    [float(row.sample_standard_deviation) for row in uncertainty], dtype=float
                )
                axis.fill_between(
                    band_x,
                    np.maximum(0.0, band_mean - band_sd),
                    np.minimum(1.0, band_mean + band_sd),
                    alpha=0.2,
                    label="Random mean ± sample SD",
                )
        axis.set_xlabel("Removed nodes / original nodes")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
        unavailable = sum(
            row.metric == metric and row.status == "unavailable" for row in results.aggregate_rows
        )
        if unavailable:
            axis.text(
                0.01,
                0.01,
                f"Unavailable aggregate points omitted: {unavailable}",
                transform=axis.transAxes,
                fontsize=8,
                verticalalignment="bottom",
            )
        _finish(figure, path)
    except Exception:
        plt.close(figure)
        raise


def plot_robustness(
    results: RobustnessResults,
    largest_component_path: str | Path,
    efficiency_path: str | Path,
) -> None:
    """Plot targeted and random perturbation trajectories with genuine uncertainty."""
    _robustness_panel(
        results,
        "largest_component_fraction",
        "Largest component / original nodes",
        "Node-removal robustness: largest component",
        largest_component_path,
    )
    _robustness_panel(
        results,
        "global_efficiency_original_node_normalized",
        "Fixed-denominator global efficiency",
        "Node-removal robustness: global efficiency",
        efficiency_path,
    )


def _primary_figure_paths(directory: Path) -> dict[str, Path]:
    return {
        "connectivity_heatmap": directory / "connectivity_matrix.png",
        "binary_adjacency": directory / "thresholded_binary_adjacency.png",
        "community_network": directory / "community_network.png",
        "hub_centrality": directory / "hub_strength.png",
    }


def _matrix_heatmap(connectome: ConnectomeData, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.1))
    try:
        if connectome.matrix_semantics == "correlation":
            image = axis.imshow(connectome.matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            scale_label = "Correlation"
        else:
            image = axis.imshow(connectome.matrix, cmap="viridis")
            scale_label = "Connection value"
        axis.set_title("Original functional connectivity matrix")
        axis.set_xlabel("ROI index (input order)")
        axis.set_ylabel("ROI index (input order)")
        figure.colorbar(image, ax=axis, shrink=0.82, label=scale_label)
        _finish(figure, path)
    except Exception:
        plt.close(figure)
        raise


def _binary_adjacency(graphs: ConstructedGraphs, connectome: ConnectomeData, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.1))
    try:
        matrix = nx.to_numpy_array(graphs.binary_graph, nodelist=connectome.roi_ids, dtype=float)
        image = axis.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.set_title("Thresholded binary adjacency")
        axis.set_xlabel("ROI index (input order)")
        axis.set_ylabel("ROI index (input order)")
        figure.colorbar(image, ax=axis, shrink=0.82, label="Retained binary edge")
        _finish(figure, path)
    except Exception:
        plt.close(figure)
        raise


def _network_positions(graph: nx.Graph, layout_seed: int) -> tuple[dict[str, np.ndarray], str]:
    coordinates = [graph.nodes[node].get("coordinates") for node in graph.nodes]
    if coordinates and all(
        isinstance(value, tuple) and len(value) == 3 and np.isfinite(value).all()
        for value in coordinates
    ):
        return (
            {
                node: np.asarray(graph.nodes[node]["coordinates"][:2], dtype=float)
                for node in graph.nodes
            },
            "ROI-coordinate x/y projection",
        )
    return (
        nx.spring_layout(graph, seed=layout_seed, weight="weight"),
        "Abstract spring layout; it is not an anatomical brain map",
    )


def _community_network(
    graphs: ConstructedGraphs,
    communities: CommunityResults,
    path: Path,
    *,
    layout_seed: int,
) -> None:
    graph = graphs.weighted_graph
    positions, layout_label = _network_positions(graph, layout_seed)
    figure, axis = plt.subplots(figsize=(8.0, 6.4))
    try:
        edges = list(graph.edges(data=True))
        subsampled = len(edges) > 600
        if subsampled:
            edges = sorted(edges, key=lambda item: float(item[2]["weight"]), reverse=True)[:600]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(source, target) for source, target, _ in edges],
            alpha=0.18,
            width=0.45,
            edge_color="#506070",
            ax=axis,
        )
        community_ids = [communities.partition[node] for node in graph.nodes]
        count = max(community_ids, default=0) + 1
        axis.scatter(
            [positions[node][0] for node in graph.nodes],
            [positions[node][1] for node in graph.nodes],
            c=community_ids,
            cmap=plt.get_cmap("tab20", max(count, 1)),
            s=18 if graph.number_of_nodes() > 100 else 42,
            linewidths=0.15,
            edgecolors="black",
        )
        if count <= 12:
            cmap = plt.get_cmap("tab20", max(count, 1))
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markerfacecolor=cmap(index),
                    markeredgecolor="black",
                    markersize=6,
                    label=f"Community {index}",
                )
                for index in range(count)
            ]
            axis.legend(handles=handles, loc="best")
        axis.set_title("Weighted Louvain community structure")
        axis.set_axis_off()
        note = layout_label
        if subsampled:
            note += "; only the 600 strongest edges are shown for visualization"
        figure.text(0.5, 0.01, note, ha="center", fontsize=8)
        _finish(figure, path)
    except Exception:
        plt.close(figure)
        raise


def _hub_strength(
    topology: TopologyResults,
    hubs: HubResults | None,
    path: Path,
) -> None:
    strength_rows = [
        row
        for row in topology.nodal_metrics
        if row.metric == "strength" and row.representation == "weighted" and row.value is not None
    ]
    strength_rows.sort(key=lambda row: (-float(row.value), row.roi_index))
    display_rows = strength_rows[: min(len(strength_rows), 30)]
    candidate_ids = set()
    if hubs is not None:
        candidate_ids = {row.roi_id for row in hubs.rows if row.is_candidate_hub}
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    try:
        labels = [row.roi_id for row in display_rows]
        values = [float(row.value) for row in display_rows]
        colours = ["#c44e52" if row.roi_id in candidate_ids else "#4c72b0" for row in display_rows]
        axis.bar(range(len(display_rows)), values, color=colours, width=0.8)
        axis.set_title("Positive node strength")
        axis.set_xlabel("ROI identifier (ranked; input-specific)")
        axis.set_ylabel("Retained positive strength")
        axis.set_xticks(range(len(display_rows)))
        axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        if candidate_ids:
            axis.text(
                0.99,
                0.98,
                "Red: candidate hub under configured graph criteria",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        if len(strength_rows) > len(display_rows):
            figure.text(0.5, 0.01, "Top 30 nodes shown.", ha="center", fontsize=8)
        _finish(figure, path)
    except Exception:
        plt.close(figure)
        raise


def write_primary_figures(
    connectome: ConnectomeData,
    graphs: ConstructedGraphs,
    directory: str | Path,
    *,
    topology: TopologyResults | None,
    communities: CommunityResults | None,
    hubs: HubResults | None,
    layout_seed: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write compact QC figures from actual run data and return created paths.

    Community and hub figures are omitted when their corresponding analysis was
    disabled or unavailable; no substitute visualization is fabricated.
    """
    output_directory = Path(directory).resolve()
    paths = _primary_figure_paths(output_directory)
    selected = {
        "connectivity_heatmap": paths["connectivity_heatmap"],
        "binary_adjacency": paths["binary_adjacency"],
    }
    if communities is not None:
        selected["community_network"] = paths["community_network"]
    if topology is not None:
        selected["hub_centrality"] = paths["hub_centrality"]
    collisions = [path for path in selected.values() if path.exists()]
    if collisions and not overwrite:
        names = ", ".join(path.name for path in collisions)
        raise OSError(f"Refusing to overwrite existing figure output(s): {names}")
    output_directory.mkdir(parents=True, exist_ok=True)
    _matrix_heatmap(connectome, selected["connectivity_heatmap"])
    _binary_adjacency(graphs, connectome, selected["binary_adjacency"])
    if communities is not None:
        _community_network(
            graphs,
            communities,
            selected["community_network"],
            layout_seed=layout_seed,
        )
    if topology is not None:
        _hub_strength(topology, hubs, selected["hub_centrality"])
    return selected

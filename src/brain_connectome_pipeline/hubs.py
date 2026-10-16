"""Transparent rank-based candidate hub characterization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from brain_connectome_pipeline.communities import CommunityNodeMetrics
from brain_connectome_pipeline.config import CommunityConfig
from brain_connectome_pipeline.metrics import TopologyResults
from brain_connectome_pipeline.models import ConnectomeData


class HubAnalysisError(ValueError):
    """Raised when hub inputs do not align or cannot be interpreted."""


@dataclass(frozen=True)
class HubNodeResult:
    roi_index: int
    roi_id: str
    label: str | None
    degree_centrality: float | None
    degree_percentile_rank: float | None
    strength: float | None
    strength_percentile_rank: float | None
    weighted_betweenness: float | None
    betweenness_percentile_rank: float | None
    weighted_participation: float | None
    participation_percentile_rank: float | None
    eligible_metric_count: int
    combined_score: float | None
    combined_rank: float | None
    eligibility_status: Literal["eligible", "ineligible"]
    eligibility_reason: str | None
    is_candidate_hub: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_index": self.roi_index,
            "roi_id": self.roi_id,
            "label": self.label,
            "degree_centrality": self.degree_centrality,
            "degree_percentile_rank": self.degree_percentile_rank,
            "strength": self.strength,
            "strength_percentile_rank": self.strength_percentile_rank,
            "weighted_betweenness": self.weighted_betweenness,
            "betweenness_percentile_rank": self.betweenness_percentile_rank,
            "weighted_participation": self.weighted_participation,
            "participation_percentile_rank": self.participation_percentile_rank,
            "eligible_metric_count": self.eligible_metric_count,
            "combined_score": self.combined_score,
            "combined_rank": self.combined_rank,
            "eligibility_status": self.eligibility_status,
            "eligibility_reason": self.eligibility_reason,
            "is_candidate_hub": self.is_candidate_hub,
        }


@dataclass(frozen=True)
class HubResults:
    rows: tuple[HubNodeResult, ...]
    status: Literal["available", "unavailable"]
    reason: str | None
    metadata: dict[str, Any]

    def summary_dict(self) -> dict[str, Any]:
        candidates = [row.roi_id for row in self.rows if row.is_candidate_hub]
        return {
            "status": self.status,
            "reason": self.reason,
            "candidate_hubs": candidates,
            "candidate_count": len(candidates),
            "eligible_nodes": sum(row.eligibility_status == "eligible" for row in self.rows),
            "metadata": self.metadata,
        }


def _average_percentile_ranks(values: dict[str, float | None]) -> dict[str, float | None]:
    available = [(node, value) for node, value in values.items() if value is not None]
    available.sort(key=lambda item: (item[1], item[0]))
    ranks: dict[str, float | None] = {node: None for node in values}
    count = len(available)
    position = 0
    while position < count:
        end = position + 1
        while end < count and available[end][1] == available[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        percentile = average_rank / count
        for index in range(position, end):
            ranks[available[index][0]] = percentile
        position = end
    return ranks


def _descending_average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for index in range(position, end):
            ranks[ordered[index][0]] = average_rank
        position = end
    return ranks


def characterize_hubs(
    topology: TopologyResults,
    community_metrics: tuple[CommunityNodeMetrics, ...],
    connectome: ConnectomeData,
    config: CommunityConfig,
) -> HubResults:
    """Aggregate four average-tie percentile ranks into candidate hub scores."""
    if not config.hub_enabled:
        raise HubAnalysisError("Hub characterization is disabled in the configuration.")
    if tuple(row.roi_id for row in community_metrics) != connectome.roi_ids:
        raise HubAnalysisError("Community metrics must follow the original ROI order.")

    topology_lookup = {
        (row.roi_id, row.metric, row.representation): row for row in topology.nodal_metrics
    }

    def topology_value(node: str, metric: str, representation: str) -> float | None:
        row = topology_lookup.get((node, metric, representation))
        if row is None or row.status != "available" or row.value is None:
            return None
        value = float(row.value)
        return value if math.isfinite(value) else None

    participation_lookup = {row.roi_id: row for row in community_metrics}
    criteria: dict[str, dict[str, float | None]] = {
        "degree": {
            node: topology_value(node, "degree_centrality", "binary") for node in connectome.roi_ids
        },
        "strength": {
            node: topology_value(node, "strength", "weighted") for node in connectome.roi_ids
        },
        "betweenness": {
            node: topology_value(node, "betweenness_centrality", "weighted")
            for node in connectome.roi_ids
        },
        "participation": {
            node: (
                participation_lookup[node].weighted_participation
                if participation_lookup[node].weighted_participation_status == "available"
                else None
            )
            for node in connectome.roi_ids
        },
    }
    ranks = {name: _average_percentile_ranks(values) for name, values in criteria.items()}
    scores: dict[str, float] = {}
    eligible_counts: dict[str, int] = {}
    for node in connectome.roi_ids:
        node_ranks = [ranks[name][node] for name in criteria if ranks[name][node] is not None]
        eligible_counts[node] = len(node_ranks)
        if len(node_ranks) >= config.hub_minimum_metrics:
            scores[node] = sum(node_ranks) / len(node_ranks)

    informative = any(
        len({value for node, value in values.items() if node in scores and value is not None}) > 1
        for values in criteria.values()
    )
    if not scores:
        status: Literal["available", "unavailable"] = "unavailable"
        reason = "No node had the configured minimum number of available hub criteria."
        combined_ranks: dict[str, float] = {}
    elif not informative:
        status = "unavailable"
        reason = "Available hub criteria have no between-node variation; ranking is uninformative."
        combined_ranks = {}
    else:
        status = "available"
        reason = None
        combined_ranks = _descending_average_ranks(scores)

    rows: list[HubNodeResult] = []
    for index, node in enumerate(connectome.roi_ids):
        eligible = node in scores
        score = scores.get(node) if status == "available" else None
        rows.append(
            HubNodeResult(
                index,
                node,
                connectome.roi_labels[index],
                criteria["degree"][node],
                ranks["degree"][node],
                criteria["strength"][node],
                ranks["strength"][node],
                criteria["betweenness"][node],
                ranks["betweenness"][node],
                criteria["participation"][node],
                ranks["participation"][node],
                eligible_counts[node],
                score,
                combined_ranks.get(node),
                "eligible" if eligible else "ineligible",
                None
                if eligible
                else (
                    f"Only {eligible_counts[node]} of 4 hub criteria were available; "
                    f"{config.hub_minimum_metrics} required."
                ),
                bool(
                    status == "available"
                    and score is not None
                    and score >= config.hub_percentile_threshold
                ),
            )
        )
    metadata = {
        "criteria": [
            "binary degree centrality",
            "positive node strength",
            "reciprocal-strength weighted betweenness",
            "weighted participation coefficient",
        ],
        "criterion_rank": "ascending average-tie percentile; higher is more prominent",
        "combined_score": "arithmetic mean of available criterion percentile ranks",
        "combined_rank": "descending average-tie rank among eligible nodes",
        "candidate_rule": (
            f"combined score >= {config.hub_percentile_threshold:.6g} with at least "
            f"{config.hub_minimum_metrics} available criteria"
        ),
        "tie_handling": "average ranks; original ROI order is the deterministic secondary order",
        "eligible_nodes": len(scores),
    }
    return HubResults(tuple(rows), status, reason, metadata)

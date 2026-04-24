"""Greedy RMSD-based pose clustering + cluster-centroid selection."""
from __future__ import annotations

import torch


def cluster_poses(
    poses: list[torch.Tensor], threshold: float = 2.0,
) -> list[list[int]]:
    """Greedy RMSD-based clustering of poses.

    Returns clusters sorted largest → smallest; each cluster is a list of
    pose indices.
    """
    n = len(poses)
    assigned = [False] * n
    clusters: list[list[int]] = []

    rmsd_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            r = (poses[i] - poses[j]).pow(2).sum(-1).mean().sqrt().item()
            rmsd_matrix[i, j] = r
            rmsd_matrix[j, i] = r

    while not all(assigned):
        best_seed = -1
        best_count = -1
        for i in range(n):
            if assigned[i]:
                continue
            count = sum(
                1 for j in range(n)
                if not assigned[j] and rmsd_matrix[i, j] < threshold
            )
            if count > best_count:
                best_count = count
                best_seed = i

        cluster = [
            j for j in range(n)
            if not assigned[j] and rmsd_matrix[best_seed, j] < threshold
        ]
        for j in cluster:
            assigned[j] = True
        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def select_by_clustering(
    poses: list[torch.Tensor], threshold: float = 2.0,
) -> int:
    """Return the pose index of the centroid of the largest cluster."""
    clusters = cluster_poses(poses, threshold)
    largest = clusters[0]

    if len(largest) == 1:
        return largest[0]

    best_idx = largest[0]
    best_mean = float("inf")
    for i in largest:
        mean_rmsd = sum(
            (poses[i] - poses[j]).pow(2).sum(-1).mean().sqrt().item()
            for j in largest if j != i
        ) / (len(largest) - 1)
        if mean_rmsd < best_mean:
            best_mean = mean_rmsd
            best_idx = i
    return best_idx


__all__ = ["cluster_poses", "select_by_clustering"]

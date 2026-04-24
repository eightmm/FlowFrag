"""Per-atom confidence feature extraction.

Library (no CLI) used by ``gen_conf_train_data.py`` and future inference
integrations.  Given a pretrained FlowFrag model and a list of candidate
poses for one complex, runs a single batched forward at t=1 with the poses
stuffed into the ligand atom/fragment nodes and returns:

  - per-atom l=0 block of the final hidden state          (``atom_scalar``)
  - per-atom channel norms of l≥1 blocks (SO(3)-invariant) (``atom_norms``)
  - per-atom Euclidean displacement from crystal           (``atom_disp``)
  - per-atom pose membership CSR pointer
  - per-pose scalar stats of v_pred / omega_pred magnitudes at t=1

Fragment (T, q) for each pose is recovered from the pose's atom positions
via per-fragment Kabsch fit, so the model's forward sees the same R_frag
injection it would at the end of ODE integration.
"""
from __future__ import annotations

import torch

from src.geometry.se3 import matrix_to_quaternion
from src.inference.sampler import build_batched_graph


# ---------------------------------------------------------------------------
# Kabsch per-fragment rigid fit
# ---------------------------------------------------------------------------
def recover_frag_state(
    atom_pos: torch.Tensor,       # [N_atom, 3] global coords (pocket-centered OK)
    local_pos: torch.Tensor,      # [N_atom, 3] fragment-local (centroid-subtracted)
    frag_id: torch.Tensor,        # [N_atom]
    n_frag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return T_frag [N_frag, 3] and q_frag [N_frag, 4] (w,x,y,z)."""
    T = torch.zeros(n_frag, 3, device=atom_pos.device, dtype=atom_pos.dtype)
    R = torch.eye(3, device=atom_pos.device, dtype=atom_pos.dtype).expand(n_frag, 3, 3).clone()
    for f in range(n_frag):
        mask = (frag_id == f)
        if mask.sum() == 0:
            continue
        y = atom_pos[mask]
        x = local_pos[mask]
        T[f] = y.mean(dim=0)
        if mask.sum() == 1:
            continue
        y_c = y - T[f]
        H = x.T @ y_c
        U, _, Vh = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vh.T @ U.T))
        D = torch.eye(3, device=atom_pos.device, dtype=atom_pos.dtype)
        D[2, 2] = d
        R[f] = Vh.T @ D @ U.T
    q = matrix_to_quaternion(R)
    return T, q


# ---------------------------------------------------------------------------
# Per-atom feature extractor (single batched forward)
# ---------------------------------------------------------------------------
def extract_per_atom_features(
    model, graph, lig_data, meta,
    raw_poses: list[torch.Tensor],
    crystal_pocket_centered: torch.Tensor,
    device: torch.device,
    t_eval: float = 1.0,
) -> dict:
    """Run model forward on B poses, return per-atom and per-pose arrays.

    Output dict (numpy arrays):
      atom_scalar     [B*N_atoms, scalar_dim]    l=0 block
      atom_norms      [B*N_atoms, D_norms]       l=1o/1e/2e/2o channel norms
      atom_disp       [B*N_atoms]                Å displacement from crystal
      atom_pose_idx   [B*N_atoms]                owning pose id (0..B-1)
      pose_rmsd       [B]                        sqrt(mean(disp^2)) per pose
      pose_n_atoms, pose_n_frags                 scalars per complex
      pose_v_mean/max/p95, pose_w_mean/max/p95   [B] each
    """
    from src.models.unified import NTYPE_LIG_ATOM

    B = len(raw_poses)
    n_frags = int(meta["num_frag"])
    n_atoms = int(meta["num_atom"])
    pocket_center = meta["pocket_center"].to(device)

    batch = build_batched_graph(graph, B, n_frags, device)
    batch["node_coords"] = batch["node_coords"] - pocket_center

    frag_sizes = lig_data["frag_sizes"].to(device)
    frag_id = lig_data["fragment_id"].to(device)
    local_pos = lig_data["frag_local_coords"].to(device)

    frag_sizes_flat = frag_sizes.repeat(B)
    frag_id_flat = (
        frag_id.repeat(B)
        + torch.arange(B, device=device).repeat_interleave(n_atoms) * n_frags
    )

    n_nodes = graph["node_coords"].shape[0]
    frag_start, frag_end = int(graph["lig_frag_slice"][0]), int(graph["lig_frag_slice"][1])
    atom_start = int(graph["lig_atom_slice"][0])
    frag_slots = torch.cat([
        torch.arange(frag_start, frag_end, device=device) + i * n_nodes for i in range(B)
    ])
    atom_slots = torch.cat([
        torch.arange(atom_start, atom_start + n_atoms, device=device) + i * n_nodes for i in range(B)
    ])

    # Per-pose (T, q) from Kabsch fit on that pose's atoms
    T_list, q_list, atom_pos_all = [], [], []
    for pose in raw_poses:
        pose_d = pose.to(device)
        T_f, q_f = recover_frag_state(pose_d, local_pos, frag_id, n_frags)
        T_list.append(T_f); q_list.append(q_f); atom_pos_all.append(pose_d)
    T_flat = torch.cat(T_list, dim=0)
    q_flat = torch.cat(q_list, dim=0)
    atom_pos_flat = torch.cat(atom_pos_all, dim=0)

    nc = batch["node_coords"].clone()
    nc[frag_slots] = T_flat
    nc[atom_slots] = atom_pos_flat
    batch["node_coords"] = nc
    batch["T_frag"] = T_flat
    batch["q_frag"] = q_flat
    batch["frag_sizes"] = frag_sizes_flat
    batch["t"] = torch.full((B, 1), t_eval, device=device, dtype=torch.float32)
    batch["frag_id_for_atoms"] = frag_id_flat

    with torch.no_grad():
        out = model(batch, return_hidden=True)

    v = out["v_pred"].view(B, n_frags, 3)
    w = out["omega_pred"].view(B, n_frags, 3)
    v_norm = v.norm(dim=-1)
    w_norm = w.norm(dim=-1)

    h = out["h"]
    D_total = h.shape[-1]
    scalar_dim = model.hidden_dim
    vec_dim = model.hidden_vec_dim
    l2_dim = model.l2_dim
    l2o_dim = model.l2o_dim

    node_type = batch["node_type"]
    h_view = h.view(B, n_nodes, D_total)
    nt_view = node_type.view(B, n_nodes)

    s_lo, s_hi = 0, scalar_dim
    v1o_lo, v1o_hi = s_hi, s_hi + vec_dim * 3
    v1e_lo, v1e_hi = v1o_hi, v1o_hi + vec_dim * 3
    l2e_lo, l2e_hi = v1e_hi, v1e_hi + l2_dim * 5
    l2o_lo, l2o_hi = l2e_hi, l2e_hi + l2o_dim * 5
    D_norms = 2 * vec_dim + l2_dim + l2o_dim

    atom_scalar_out = torch.zeros(B * n_atoms, scalar_dim, device=device)
    atom_norms_out = torch.zeros(B * n_atoms, D_norms, device=device)
    atom_pose_idx = torch.arange(B, device=device).repeat_interleave(n_atoms)

    crystal_dev = crystal_pocket_centered.to(device)
    atom_disp = torch.zeros(B * n_atoms, device=device)

    for b in range(B):
        h_b = h_view[b]
        atom_mask = (nt_view[b] == NTYPE_LIG_ATOM)
        h_atom = h_b[atom_mask]
        assert h_atom.shape[0] == n_atoms

        atom_scalar_out[b * n_atoms : (b + 1) * n_atoms] = h_atom[:, s_lo:s_hi]

        v1o = h_atom[:, v1o_lo:v1o_hi].reshape(n_atoms, vec_dim, 3).norm(dim=-1)
        v1e = h_atom[:, v1e_lo:v1e_hi].reshape(n_atoms, vec_dim, 3).norm(dim=-1)
        parts = [v1o, v1e]
        if l2_dim:
            parts.append(h_atom[:, l2e_lo:l2e_hi].reshape(n_atoms, l2_dim, 5).norm(dim=-1))
        if l2o_dim:
            parts.append(h_atom[:, l2o_lo:l2o_hi].reshape(n_atoms, l2o_dim, 5).norm(dim=-1))
        atom_norms_out[b * n_atoms : (b + 1) * n_atoms] = torch.cat(parts, dim=-1)

        pose = raw_poses[b].to(device)
        atom_disp[b * n_atoms : (b + 1) * n_atoms] = (pose - crystal_dev).norm(dim=-1)

    pose_rmsd = atom_disp.view(B, n_atoms).pow(2).mean(dim=1).sqrt()
    v_mean = v_norm.mean(dim=1)
    v_max = v_norm.amax(dim=1)
    v_p95 = v_norm.quantile(0.95, dim=1) if n_frags >= 5 else v_norm.amax(dim=1)
    w_mean = w_norm.mean(dim=1)
    w_max = w_norm.amax(dim=1)
    w_p95 = w_norm.quantile(0.95, dim=1) if n_frags >= 5 else w_norm.amax(dim=1)

    return {
        "atom_scalar": atom_scalar_out.cpu().numpy(),
        "atom_norms": atom_norms_out.cpu().numpy(),
        "atom_disp": atom_disp.cpu().numpy(),
        "atom_pose_idx": atom_pose_idx.cpu().numpy(),
        "pose_rmsd": pose_rmsd.cpu().numpy(),
        "pose_n_atoms": n_atoms,
        "pose_n_frags": n_frags,
        "pose_v_mean": v_mean.cpu().numpy(),
        "pose_v_max": v_max.cpu().numpy(),
        "pose_v_p95": v_p95.cpu().numpy(),
        "pose_w_mean": w_mean.cpu().numpy(),
        "pose_w_max": w_max.cpu().numpy(),
        "pose_w_p95": w_p95.cpu().numpy(),
    }


__all__ = ["extract_per_atom_features", "recover_frag_state"]

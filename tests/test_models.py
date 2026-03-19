"""Tests for model architecture (requires GPU / cuEquivariance)."""

import pytest
import torch
from torch_geometric.data import HeteroData

_HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required for cuEquivariance")


def _make_batch(prot_pos, atom_pos, frag_T, device, q_frag=None):
    """Build a minimal HeteroData batch for FlowFrag."""
    data = HeteroData()
    n_prot = prot_pos.shape[0]
    n_atom = atom_pos.shape[0]

    data["protein"].pos = prot_pos.to(device)
    data["protein"].x = torch.arange(n_prot, device=device) % 20

    data["atom"].x = torch.arange(n_atom, device=device) % 13
    data["atom"].charge = torch.zeros(n_atom, device=device)
    data["atom"].aromatic = torch.zeros(n_atom, dtype=torch.long, device=device)
    data["atom"].hybridization = torch.ones(n_atom, dtype=torch.long, device=device)
    data["atom"].in_ring = torch.zeros(n_atom, dtype=torch.long, device=device)
    data["atom"].pos = atom_pos.to(device)
    data["atom"].pos_t = atom_pos.to(device)
    data["atom"].fragment_id = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], device=device
    )

    data["atom", "bond", "atom"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 3, 4, 6, 7, 8, 9]], device=device
    )
    bond_attr = torch.stack(
        [
            torch.tensor([1, 1, 2, 1, 1, 1, 2, 1]),
            torch.tensor([0, 1, 0, 0, 0, 1, 0, 0]),
            torch.tensor([0, 1, 0, 0, 0, 1, 0, 0]),
        ],
        dim=-1,
    ).to(device)
    data["atom", "bond", "atom"].edge_attr = bond_attr
    data["atom", "cut", "atom"].edge_index = torch.tensor(
        [[4, 5], [5, 4]], device=device
    )

    data["fragment"].T_frag = frag_T.to(device)
    if q_frag is None:
        q_frag = torch.tensor([[0.9239, 0.3827, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0]])
    data["fragment"].q_frag = q_frag.to(device)
    data["fragment"].size = torch.tensor([5, 5], device=device)
    data.t = torch.tensor([0.4], device=device)
    return data


@requires_cuda
def test_flowfrag_forward_shape():
    """Forward pass produces correct output shapes."""
    from src.models import FlowFrag

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = FlowFrag(
        hidden_dim=64,
        num_encoder_layers_prot=2,
        num_encoder_layers_lig=2,
        num_docking_layers=2,
        hidden_scalar_dim=64,
        hidden_vec_dim=16,
    ).to(device)
    model.eval()

    batch = _make_batch(torch.randn(8, 3), torch.randn(10, 3), torch.randn(2, 3), device)

    with torch.no_grad():
        out = model(batch)

    assert out["v_pred"].shape == (2, 3)
    assert out["omega_pred"].shape == (2, 3)
    assert not torch.isnan(out["v_pred"]).any()
    assert not torch.isnan(out["omega_pred"]).any()


@requires_cuda
def test_cross_topology_full_edges_respect_batch():
    """Full coarse topology should connect all pairs only within each batch item."""
    from src.models.docking_head import _build_cross_topology_edges

    device = torch.device("cuda")
    src_pos = torch.randn(3, 3, device=device)
    dst_pos = torch.randn(2, 3, device=device)
    src_batch = torch.tensor([0, 0, 1], device=device)
    dst_batch = torch.tensor([0, 1], device=device)

    src, dst = _build_cross_topology_edges(
        src_pos,
        dst_pos,
        topology="full",
        r=1.0,
        max_neighbors=1,
        src_batch=src_batch,
        dst_batch=dst_batch,
    )

    pairs = sorted(zip(src.cpu().tolist(), dst.cpu().tolist()))
    assert pairs == [(0, 0), (1, 0), (2, 1)]


@requires_cuda
def test_flowfrag_forward_shape_with_global_coarse_edges():
    """Forward pass works when protein-fragment and fragment graphs are dense."""
    from src.models import FlowFrag

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = FlowFrag(
        hidden_dim=64,
        num_encoder_layers_prot=2,
        num_encoder_layers_lig=2,
        num_docking_layers=2,
        hidden_scalar_dim=64,
        hidden_vec_dim=16,
        pf_topology="full",
        ff_topology="full",
    ).to(device)
    model.eval()

    batch = _make_batch(torch.randn(8, 3), torch.randn(10, 3), torch.randn(2, 3), device)

    with torch.no_grad():
        out = model(batch)

    assert out["v_pred"].shape == (2, 3)
    assert out["omega_pred"].shape == (2, 3)


@requires_cuda
def test_flowfrag_forward_shape_with_global_coarse_and_cut_bond_edges():
    """Forward pass works with dense coarse graphs plus explicit cut-bond edges."""
    from src.models import FlowFrag

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = FlowFrag(
        hidden_dim=64,
        num_encoder_layers_prot=2,
        num_encoder_layers_lig=2,
        num_docking_layers=2,
        hidden_scalar_dim=64,
        hidden_vec_dim=16,
        pf_topology="full",
        ff_topology="full",
        use_cut_bond_edges=True,
    ).to(device)
    model.eval()

    batch = _make_batch(torch.randn(8, 3), torch.randn(10, 3), torch.randn(2, 3), device)

    with torch.no_grad():
        out = model(batch)

    assert out["v_pred"].shape == (2, 3)
    assert out["omega_pred"].shape == (2, 3)


@requires_cuda
def test_flowfrag_se3_equivariance():
    """v_pred is SE(3)-equivariant under rotation of all 3D inputs.

    omega_pred is not tested for equivariance because it uses an analytic
    formula based on q_t (scalar-gated axis-angle) which is correct by
    construction within the q_1=I convention, but does not transform
    covariantly under global rotations of positions alone.
    """
    from scipy.spatial.transform import Rotation

    from src.models import FlowFrag

    device = torch.device("cuda")
    torch.manual_seed(42)

    model = FlowFrag(
        hidden_dim=64,
        num_encoder_layers_prot=2,
        num_encoder_layers_lig=2,
        num_docking_layers=2,
        hidden_scalar_dim=64,
        hidden_vec_dim=16,
    ).to(device)
    model.eval()

    prot_pos = torch.randn(8, 3)
    atom_pos = torch.randn(10, 3)
    frag_T = torch.randn(2, 3)

    R = torch.from_numpy(Rotation.random(random_state=123).as_matrix()).float().to(device)

    # Convert R to quaternion for rotating q_frag
    from src.geometry.se3 import matrix_to_quaternion, quaternion_multiply
    R_quat = matrix_to_quaternion(R.unsqueeze(0))  # [1, 4]

    q_frag = torch.tensor([[0.9239, 0.3827, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0]])
    q_frag_rot = quaternion_multiply(R_quat.cpu().expand(2, -1), q_frag)

    # Original forward
    batch1 = _make_batch(prot_pos, atom_pos, frag_T, device, q_frag=q_frag)
    with torch.no_grad():
        out1 = model(batch1)

    # Rotated forward (all 3D inputs + quaternions rotated)
    batch2 = _make_batch(prot_pos @ R.cpu().T, atom_pos @ R.cpu().T, frag_T @ R.cpu().T, device, q_frag=q_frag_rot)
    with torch.no_grad():
        out2 = model(batch2)

    # v (1o) should rotate under SO(3)
    v1_rot = out1["v_pred"] @ R.T
    torch.testing.assert_close(out2["v_pred"], v1_rot, atol=1e-4, rtol=1e-4)

    # omega equivariance not tested — analytic omega uses axis_angle(q_t)
    # which is correct by construction but not equivariant under global rotation.


@requires_cuda
def test_single_atom_fragment_omega_zero():
    """Single-atom fragments should have omega_pred = 0."""
    from src.models import FlowFrag

    device = torch.device("cuda")
    torch.manual_seed(0)

    model = FlowFrag(
        hidden_dim=64,
        num_encoder_layers_prot=2,
        num_encoder_layers_lig=2,
        num_docking_layers=2,
        hidden_scalar_dim=64,
        hidden_vec_dim=16,
    ).to(device)
    model.eval()

    data = HeteroData()
    data["protein"].pos = torch.randn(5, 3, device=device)
    data["protein"].x = torch.arange(5, device=device) % 20

    data["atom"].x = torch.tensor([0, 1, 2], device=device)
    data["atom"].charge = torch.zeros(3, device=device)
    data["atom"].aromatic = torch.zeros(3, dtype=torch.long, device=device)
    data["atom"].hybridization = torch.ones(3, dtype=torch.long, device=device)
    data["atom"].in_ring = torch.zeros(3, dtype=torch.long, device=device)
    data["atom"].pos = torch.randn(3, 3, device=device)
    data["atom"].pos_t = torch.randn(3, 3, device=device)
    # frag 1 = single atom
    data["atom"].fragment_id = torch.tensor([0, 0, 1], device=device)

    data["atom", "bond", "atom"].edge_index = torch.tensor([[0, 1], [1, 2]], device=device)
    data["atom", "bond", "atom"].edge_attr = torch.tensor(
        [[1, 0, 0], [1, 0, 0]], device=device
    )

    data["fragment"].T_frag = torch.randn(2, 3, device=device)
    # Single-atom frag gets identity quaternion, multi-atom gets non-trivial rotation
    q = torch.tensor([[0.9239, 0.3827, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
    data["fragment"].q_frag = q
    data["fragment"].size = torch.tensor([2, 1], device=device)
    data.t = torch.tensor([0.5], device=device)

    with torch.no_grad():
        out = model(data)

    # Fragment 1 (single atom) should have zero angular velocity
    assert torch.allclose(out["omega_pred"][1], torch.zeros(3, device=device), atol=1e-8)

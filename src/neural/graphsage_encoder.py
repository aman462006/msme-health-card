"""
GraphSAGE encoder for buyer-supplier e-way-bill network.

PDF: "GraphSAGE on e-way-bill buyer-supplier graph -> 16-dim node embedding
capturing supply-chain centrality and network default contagion."

In prototype: if no graph data is available the encoder returns zeros.
When e-way-bill data arrives via GSTN API, pass an adjacency list and
this module computes the embedding without any code changes.
"""
import numpy as np
from dataclasses import dataclass

GRAPH_EMB_DIM = 16


@dataclass
class NodeFeatures:
    gstin: str
    total_supply_value: float   # from e-way bills (supplier side)
    total_receive_value: float  # from e-way bills (buyer side)
    unique_buyers: int
    unique_suppliers: int
    network_degree: int


def _mean_aggregate(neighbor_embs: list[np.ndarray]) -> np.ndarray:
    if not neighbor_embs:
        return np.zeros(GRAPH_EMB_DIM, dtype=np.float32)
    return np.mean(neighbor_embs, axis=0).astype(np.float32)


def encode_node(
    node: NodeFeatures,
    neighbor_features: list[NodeFeatures] | None = None,
) -> np.ndarray:
    """
    Two-layer GraphSAGE mean aggregation (no learned weights in prototype).
    Layer 1: aggregate neighbor raw features.
    Layer 2: concat self + aggregated -> project to 16-dim via fixed linear map.

    Returns 16-dim embedding. Falls back to zeros if node has no data.
    """
    if node.total_supply_value == 0 and node.total_receive_value == 0:
        return np.zeros(GRAPH_EMB_DIM, dtype=np.float32)

    self_feats = np.array([
        np.log1p(node.total_supply_value),
        np.log1p(node.total_receive_value),
        float(node.unique_buyers),
        float(node.unique_suppliers),
        float(node.network_degree),
        node.total_supply_value / (node.total_receive_value + 1.0),
    ], dtype=np.float32)

    if neighbor_features:
        neighbor_vecs = []
        for n in neighbor_features:
            neighbor_vecs.append(np.array([
                np.log1p(n.total_supply_value),
                np.log1p(n.total_receive_value),
                float(n.unique_buyers),
                float(n.unique_suppliers),
                float(n.network_degree),
                n.total_supply_value / (n.total_receive_value + 1.0),
            ], dtype=np.float32))
        agg = _mean_aggregate(neighbor_vecs)
        combined = np.concatenate([self_feats, agg])  # 12-dim
    else:
        combined = np.concatenate([self_feats, np.zeros(6, dtype=np.float32)])

    # Fixed deterministic projection matrix seeded by feature positions
    rng = np.random.default_rng(seed=42)
    W = rng.standard_normal((GRAPH_EMB_DIM, combined.shape[0])).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-6
    emb = np.tanh(W @ combined)
    return emb


def encode_from_eway_bills(gstin: str, eway_records: list[dict]) -> np.ndarray:
    """
    Builds a NodeFeatures from raw e-way bill records and returns embedding.
    Each record should have: {'supplier_gstin', 'buyer_gstin', 'value'}.
    """
    if not eway_records:
        return np.zeros(GRAPH_EMB_DIM, dtype=np.float32)

    supply_value = sum(r["value"] for r in eway_records if r.get("supplier_gstin") == gstin)
    receive_value = sum(r["value"] for r in eway_records if r.get("buyer_gstin") == gstin)
    buyers = {r["buyer_gstin"] for r in eway_records if r.get("supplier_gstin") == gstin}
    suppliers = {r["supplier_gstin"] for r in eway_records if r.get("buyer_gstin") == gstin}

    node = NodeFeatures(
        gstin=gstin,
        total_supply_value=supply_value,
        total_receive_value=receive_value,
        unique_buyers=len(buyers),
        unique_suppliers=len(suppliers),
        network_degree=len(buyers) + len(suppliers),
    )
    return encode_node(node)

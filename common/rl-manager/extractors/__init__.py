from extractors.euromlsys_extractor import CustomFeatureExtractor
from extractors.attention_extractor import AttentionPoolingFeatureExtractor
from extractors.hybrid_extractor import HybridPoolingExtractor
from extractors.hybrid_pre_head_extractor import HybridPoolingPreHeadExtractor
from extractors.type_stratified_extractor import TypeStratifiedExtractor
from extractors.type_stratified_pre_head_extractor import TypeStratifiedPreHeadExtractor
from extractors.rbf_extractor import RBFPoolingExtractor
from extractors.type_stratified_embed_extractor import TypeStratifiedEmbedExtractor
from extractors.fusion_extractor import FusionMLPExtractor
from extractors.attention_idfree_extractor import IDFreeAttentionExtractor
from extractors.pma_extractor import PMAPoolingExtractor
from extractors.hierarchical_extractor import HierarchicalJobDCExtractor
from extractors.hybrid_rbf_extractor import HybridRBFPoolExtractor
from extractors.swat_extractor import SWATExtractor
from extractors.aria_extractor import ARIAExtractor
from extractors.tsar_extractor import TSARExtractor
from extractors.deepsets_extractor import DeepSetsExtractor
from extractors.token_encoder import TokenEncoder
from extractors.pointer_policy import PointerPolicy, PositionalHeadPolicy
from extractors.featurize import HOST_FEATURES, JOB_FEATURES

EXTRACTOR_REGISTRY = {
    "euromlsys": CustomFeatureExtractor,
    "custom": CustomFeatureExtractor,  # backward-compat alias
    "attention": AttentionPoolingFeatureExtractor,  # alias; mean-pool variant removed
    "attention_pooling": AttentionPoolingFeatureExtractor,
    # A3: SPANE's shared per-machine embedding and advantage module (the pointer head).
    "spane": TokenEncoder,
    "hybrid": HybridPoolingExtractor,
    "hybrid_pre_head": HybridPoolingPreHeadExtractor,
    "type_stratified": TypeStratifiedExtractor,
    "type_stratified_pre_head": TypeStratifiedPreHeadExtractor,
    "rbf": RBFPoolingExtractor,
    "type_stratified_embed": TypeStratifiedEmbedExtractor,
    "fusion": FusionMLPExtractor,
    "attention_idfree": IDFreeAttentionExtractor,
    "pma": PMAPoolingExtractor,
    "hierarchical": HierarchicalJobDCExtractor,
    "hybrid_rbf": HybridRBFPoolExtractor,
    "swat": SWATExtractor,
    "aria": ARIAExtractor,
    "tsar": TSARExtractor,
    "deepsets": DeepSetsExtractor,
    "a5": TokenEncoder,
    "a5_positional_head": TokenEncoder,     # ablation V1: A5's tokens, a positional head
    "a5_unmasked_pool": TokenEncoder,       # ablation V2
    "a5_scalar_dc_type": TokenEncoder,      # ablation V3
    "a5_dc_id_embedding": TokenEncoder,     # ablation V4
    "a5_no_reach": TokenEncoder,            # ablation V5
}


def _register_turret() -> None:
    try:
        from extractors.turret_extractor import TurretGNNExtractor
        EXTRACTOR_REGISTRY["turret"] = TurretGNNExtractor
    except ImportError:
        pass  # torch_geometric not installed; turret unavailable


_register_turret()


# Architectures built on TokenEncoder: whether the encoder uses job <-> DC cross-attention.
TOKEN_ENCODER_CROSS_ATTENTION = {
    "a5": True, "a5_positional_head": True, "a5_unmasked_pool": True, "a5_scalar_dc_type": True,
    "a5_dc_id_embedding": True, "a5_no_reach": True, "spane": False,
}
# The ablation flag each A5 variant turns on (TokenEncoder kwargs).
TOKEN_ENCODER_ABLATION = {"a5_unmasked_pool": "unmasked_pool", "a5_scalar_dc_type": "scalar_dc_type",
                          "a5_dc_id_embedding": "dc_id_embedding", "a5_no_reach": "no_reach"}
# Their action heads: A3, A5 and V2-V5 share the pointer head; V1 is positional.
TOKEN_HEAD = {name: PointerPolicy for name in TOKEN_ENCODER_CROSS_ATTENTION}
TOKEN_HEAD["a5_positional_head"] = PositionalHeadPolicy


def get_policy_class(name: str, default):
    """The TokenEncoder architectures act through a head on the tokens (TOKEN_HEAD); the others,
    A4 (TURRET's output network on its readout) included, keep SB3's head."""
    return TOKEN_HEAD.get(name, default)


def build_policy_head_kwargs(name: str, params: dict) -> dict:
    return {"head_dim": params.get("head_dim", 64)} if name in TOKEN_HEAD else {}


def get_extractor_class(name: str):
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown feature extractor: '{name}'. "
            f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
        )
    cls = EXTRACTOR_REGISTRY[name]
    # Extractors written for an older observation layout would reshape the flat arrays
    # with the wrong stride and misread every slot without raising (960 = 192*5 = 320*3).
    host_dim = getattr(cls, "HOST_FEAT_DIM", HOST_FEATURES)
    job_dim = getattr(cls, "JOB_FEAT_DIM", JOB_FEATURES)
    if (host_dim, job_dim) != (HOST_FEATURES, JOB_FEATURES):
        raise ValueError(
            f"Feature extractor '{name}' reads {host_dim} host / {job_dim} job features per "
            f"slot, but the observation has {HOST_FEATURES} / {JOB_FEATURES}; port it first."
        )
    return cls


def build_extractor_kwargs(name: str, params: dict) -> dict:
    """Build the features_extractor_kwargs dict for the chosen extractor."""
    kwargs: dict = {"features_dim": params.get("features_dim", 64)}

    if name in ("euromlsys", "custom"):
        kwargs.update({
            "embedding_size": params.get("embedding_size", 32),
            "hidden_dim": params.get("hidden_dim", 128),
            "adaptation_bottleneck": params.get("adaptation_bottleneck", False),
            "use_residual": params.get("use_residual", True),
            "dropout": params.get("dropout", 0.1),
        })
    elif name == "turret":
        kwargs.update({
            "gnn_hidden": params.get("gnn_hidden", 64),
            "gnn_heads": params.get("gnn_heads", 4),
            "num_layers": params.get("num_layers", 2),
            "dropout": params.get("dropout", 0.1),
        })
    elif name == "deepsets":
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 128),
        })
    elif name in TOKEN_ENCODER_CROSS_ATTENTION:
        kwargs = {
            "token_dim": params.get("token_dim", 64),
            "cross_attention": TOKEN_ENCODER_CROSS_ATTENTION[name],
        }
        if name in TOKEN_ENCODER_ABLATION:
            kwargs[TOKEN_ENCODER_ABLATION[name]] = True
    elif name in ("attention", "attention_pooling"):
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_datacenter_types", params.get("max_dc_types", 3)),
        })
    elif name in ("hybrid", "hybrid_pre_head"):
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 64),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "n_heads": params.get("n_heads", 4),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
        })
    elif name in ("type_stratified", "type_stratified_pre_head"):
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 32),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
            "max_datacenters": params.get("max_datacenters", 8),
        })
    elif name == "rbf":
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 64),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "max_datacenters": params.get("max_datacenters", 8),
        })
    elif name == "type_stratified_embed":
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 32),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "dc_type_emb_dim": params.get("dc_type_emb_dim", 16),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
            "max_datacenters": params.get("max_datacenters", 8),
        })
    elif name == "fusion":
        kwargs.update({
            "dc_type_emb_dim": params.get("dc_type_emb_dim", 16),
            "hidden_dim": params.get("hidden_dim", 128),
            "dropout": params.get("dropout", 0.1),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })
    elif name == "attention_idfree":
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })
    elif name == "pma":
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 64),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "n_heads": params.get("n_heads", 4),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })
    elif name == "hierarchical":
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })
    elif name == "hybrid_rbf":
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 64),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
        })
    elif name == "swat":
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })
    elif name == "aria":
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })
    elif name == "tsar":
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_dc_types", params.get("max_datacenter_types", 3)),
        })

    return kwargs

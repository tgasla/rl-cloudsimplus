from extractors.euromlsys_extractor import CustomFeatureExtractor
from extractors.attention_extractor import AttentionPoolingFeatureExtractor
from extractors.spane_extractor import SPANEFeatureExtractor

EXTRACTOR_REGISTRY = {
    "euromlsys": CustomFeatureExtractor,
    "custom": CustomFeatureExtractor,  # backward-compat alias
    "attention": AttentionPoolingFeatureExtractor,  # alias; mean-pool variant removed
    "attention_pooling": AttentionPoolingFeatureExtractor,
    "spane": SPANEFeatureExtractor,
}


def _register_turret() -> None:
    try:
        from extractors.turret_extractor import TurretGNNExtractor
        EXTRACTOR_REGISTRY["turret"] = TurretGNNExtractor
    except ImportError:
        pass  # torch_geometric not installed; turret unavailable


_register_turret()


def get_extractor_class(name: str):
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown feature extractor: '{name}'. "
            f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
        )
    return EXTRACTOR_REGISTRY[name]


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
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_datacenter_types", params.get("max_dc_types", 3)),
        })
    elif name in ("attention", "attention_pooling"):
        kwargs.update({
            "hidden_dim": params.get("hidden_dim", 64),
            "n_heads": params.get("n_heads", 4),
            "n_layers": params.get("n_layers", 2),
            "dropout": params.get("dropout", 0.1),
            "max_datacenters": params.get("max_datacenters", 8),
            "max_dc_types": params.get("max_datacenter_types", params.get("max_dc_types", 3)),
        })
    elif name == "spane":
        kwargs.update({
            "dc_emb_dim": params.get("dc_emb_dim", 64),
            "job_emb_dim": params.get("job_emb_dim", 64),
            "hidden_dim": params.get("hidden_dim", 128),
            "max_datacenters": params.get("max_datacenters", 8),
        })

    return kwargs

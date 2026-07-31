"""Model registry.

Seven builders ship here.  Two families that appear in the published results,
``hrformer-b`` and ``transpose-b``, are deliberately **not** included: the parameter
audit of the original builders reports 9,536 and 9,600 trainable parameters, three to
four orders of magnitude below the published HRFormer-B and TransPose-B
architectures, so those builders did not instantiate the networks their labels claim.
Their rows remain in ``results/runs.jsonl`` for completeness and are flagged in
``docs/limitations.md``; shipping the builders would invite a reader to treat them as
faithful implementations.  See ``docs/adr/0002-exclude-unverified-backbones.md``.

Importing this module does not import ``torch``: :func:`available_models` reads a
plain tuple, and :func:`build_model` pulls the backbones in on demand.
"""

from __future__ import annotations

from typing import Any

from vitvert.models.names import LARGE_VARIANTS, MODEL_NAMES, available_models

__all__ = [
    "LARGE_VARIANTS",
    "MODEL_NAMES",
    "available_models",
    "build_model",
]


def build_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered model. Requires ``torch`` and ``transformers``.

    Args:
        name: one of :func:`available_models`.
        **kwargs: forwarded to the backbone (``num_keypoints``, ``pretrained``, ...).

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if name not in MODEL_NAMES:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(available_models())}")

    from vitvert.models.backbones import PlainViT, ViTPose

    if name == "vit-base":
        return PlainViT(**kwargs)
    return ViTPose(name, **kwargs)


def __getattr__(item: str) -> Any:
    """Expose the torch-dependent symbols lazily."""
    if item in {"PlainViT", "ViTPose", "VITPOSE_SPECS", "BackboneSpec"}:
        from vitvert.models import backbones

        return getattr(backbones, item)
    if item in {"KeypointHead", "ClassificationHead"}:
        from vitvert.models import heads

        return getattr(heads, item)
    raise AttributeError(f"module {__name__!r} has no attribute {item!r}")

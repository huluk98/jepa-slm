"""Utilities for JEPA-augmented small language model research."""

__all__ = [
    "JepaEncoderDecoder",
    "JepaSettings",
    "ModelShape",
    "TrainingConfig",
    "TrainingObjective",
    "assert_encoder_only_jepa_contract",
    "load_training_config",
]


def __getattr__(name: str):
    if name in {"JepaSettings", "ModelShape", "TrainingConfig", "TrainingObjective", "load_training_config"}:
        from . import config

        return getattr(config, name)
    if name in {"JepaEncoderDecoder", "assert_encoder_only_jepa_contract"}:
        from . import modeling

        return getattr(modeling, name)
    raise AttributeError(name)

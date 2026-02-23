"""
registry.py
===========
Central registry that maps string identifiers → concrete classes.

Usage in experiment configs:
    model:
      class: DiffWaveDiffusionModule
      params: { ... }

The runner calls  ``registry.get("DiffWaveDiffusionModule")``  to obtain the
class object, then instantiates it with the supplied params.

Design notes
------------
* **Lazy imports** – classes are imported only when first requested.  This
  avoids import-time errors when a dependency is not installed (e.g. you don't
  need TimeGAN deps when running DiffWave).
* **User-extensible** – call ``registry.register("MyClass", MyClass)`` or use
  the ``@registry.auto_register`` decorator before the runner starts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type


# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type | Callable[..., Any]] = {}

# Lazy-import map: "ClassName" -> ("module.path", "ClassName")
_LAZY_MAP: Dict[str, tuple[str, str]] = {
    # --- Datasets ---
    "AZT1D2025Dataset":              ("data_prep", "AZT1D2025Dataset"),
    "BrisT1DDataset":                ("data_prep", "BrisT1DDataset"),
    "HUPA_UCMDataset":               ("data_prep", "HUPA_UCMDataset"),
    # --- Data Management ---
    "DataSplitter":                  ("data_management.splitter", "DataSplitter"),
    "MinMaxNormalizer":              ("data_management.normalization", "MinMaxNormalizer"),
    "QuantileNormalizer":            ("data_management.normalization", "MinMaxNormalizer"),
    # --- Windowing ---
    "WindowBuilder":                 ("windowing", "WindowBuilder"),
    "FullSequenceBuilder":           ("windowing", "FullSequenceBuilder"),
    "ConditionalWindowPack":         ("windowing", "ConditionalWindowPack"),
    # --- Reconstruction ---
    "ReconstructionConfig":          ("reconstruction", "ReconstructionConfig"),
    "WindowReconstructor":           ("reconstruction", "WindowReconstructor"),
    "FullSequenceReconstructor":     ("reconstruction", "FullSequenceReconstructor"),
    # --- Models ---
    "ConditionalTimeGanModule":      ("models", "ConditionalTimeGanModule"),
    "StaticConditionalTimeGanModule":("models", "StaticConditionalTimeGanModule"),
    "ProjectedStaticTimeGanModule":  ("models", "ProjectedStaticTimeGanModule"),

    "DiffWaveDiffusionModule":       ("models", "DiffWaveDiffusionModule"),
    # --- Training ---
    "Trainer":                       ("training", "Trainer"),
    "WandBLogger":                   ("training.loggers", "WandBLogger"),
    "GenerativeVisualizer":          ("training.callbacks", "GenerativeVisualizer"),
    "GenerativeMomentsMetric":       ("training.callbacks", "GenerativeMomentsMetric"),
    "GenerativePCAVisualizer":       ("training.callbacks", "GenerativePCAVisualizer"),
    # --- Inference ---
    "InferenceOrchestrator":         ("inference", "InferenceOrchestrator"),
    "SequenceInferenceOrchestrator": ("inference", "SequenceInferenceOrchestrator"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(name: str, cls: Type | Callable[..., Any]) -> None:
    """Manually register a class/factory under *name*."""
    _REGISTRY[name] = cls


def auto_register(cls: Type) -> Type:
    """Decorator – registers the class under its own ``__name__``."""
    _REGISTRY[cls.__name__] = cls
    return cls


def get(name: str) -> Type | Callable[..., Any]:
    """Return the class registered under *name*, lazy-importing if needed."""
    if name in _REGISTRY:
        return _REGISTRY[name]

    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, attr)
        _REGISTRY[name] = cls          # cache for next lookup
        return cls

    raise KeyError(
        f"'{name}' is not registered and has no lazy-import entry. "
        f"Available: {sorted(set(list(_REGISTRY) + list(_LAZY_MAP)))}"
    )


def instantiate(class_name: str, **kwargs) -> Any:
    """Shortcut: look up *class_name* and call it with **kwargs**."""
    cls = get(class_name)
    return cls(**kwargs)


def available() -> list[str]:
    """Return all registered + lazy-importable names."""
    return sorted(set(list(_REGISTRY) + list(_LAZY_MAP)))
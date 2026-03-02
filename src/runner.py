"""
runner.py
=========
Config-driven experiment runner.

Takes a single experiment configuration (dict or YAML path) and executes the
full pipeline:  data → split → normalize → window → train → infer.

Every concrete class is resolved through the **registry**, so adding a new
model/dataset/callback only requires a registry entry – no runner edits.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any
import inspect

import yaml
import torch

import registry
from config_utils import deep_merge
from feature_utils import resolve_features


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_global_config(
    cfg_section: Any,
    experiment_root: Path,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """
    Global config can be:
      - a file path  (str)          → load YAML, then apply overrides
      - an inline dict              → use as-is (overrides still applied)
      - missing / None              → empty dict (overrides still applied)

    If *overrides* is provided, it is deep-merged on top of the resolved base.
    """
    if cfg_section is None:
        base = {}
    elif isinstance(cfg_section, dict):
        base = copy.deepcopy(cfg_section)
    else:
        # Treat as a path (relative paths resolved against experiment_root)
        p = Path(cfg_section)
        if not p.is_absolute():
            p = experiment_root / p
        base = _load_yaml(p)

    if overrides:
        base = deep_merge(base, overrides)

    return base


def _make_path(value: Any, experiment_root: Path) -> Path:
    """Resolve a path string relative to *experiment_root*."""
    p = Path(value)
    if not p.is_absolute():
        p = experiment_root / p
    return p


def _instantiate_from_cfg(
    cfg: dict[str, Any],
    *,
    extra_kwargs: dict[str, Any] | None = None,
    experiment_root: Path | None = None,
) -> Any:
    """
    Instantiate a component from a config block of the form::

        class: ClassName
        params:
          key: value
          ...

    *extra_kwargs* are merged into params (params takes precedence).
    Path-valued params whose keys end with ``_path``, ``_root``, ``_dir``, or
    ``_file`` are resolved relative to *experiment_root*.
    """
    cls_name = cfg["class"]
    params = copy.deepcopy(cfg.get("params", {}))

    # Resolve path-like params
    if experiment_root is not None:
        path_suffixes = ("_path", "_root", "_dir", "_file")
        for k, v in list(params.items()):
            if isinstance(v, str) and any(k.endswith(s) for s in path_suffixes):
                params[k] = _make_path(v, experiment_root)

    if extra_kwargs:
        for k, v in extra_kwargs.items():
            params.setdefault(k, v)

    return registry.instantiate(cls_name, **params)


def _build_callbacks(
    cb_cfgs: list[dict[str, Any]] | None,
    *,
    fixed_batch: Any = None,
    device: torch.device,
) -> list:
    """Instantiate a list of callback configs, injecting common dependencies."""
    if not cb_cfgs:
        return []
    cbs = []
    for cb_cfg in cb_cfgs:
        cls = registry.get(cb_cfg["class"])
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        extra: dict[str, Any] = {}
        if "device" in params:
            extra["device"] = device
        if "fixed_batch" in params and fixed_batch is not None:
            extra["fixed_batch"] = fixed_batch

        cbs.append(_instantiate_from_cfg(cb_cfg, extra_kwargs=extra))
    return cbs


# ═══════════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: dict[str, Any] | str | Path) -> Path:
    """
    Execute a single experiment described by *config*.

    Parameters
    ----------
    config : dict | str | Path
        Either an already-loaded config dict, or a path to a YAML file.

    Returns
    -------
    Path
        The output directory for this experiment.
    """

    # ------------------------------------------------------------------
    # 0. LOAD & RESOLVE CONFIG
    # ------------------------------------------------------------------
    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve()
        experiment_root = config_path.parent
        config = _load_yaml(config_path)
    else:
        experiment_root = Path.cwd()

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Runner] Device: {device}")

    # Global config: file path OR inline dict, with optional overrides
    global_config = _resolve_global_config(
        config.get("global_config"),
        experiment_root,
        overrides=config.get("global_config_overrides"),
    )
    schema = global_config.get("schema", {})
    target_col = schema.get("target_col", config.get("target_col", "glucose"))

    # Output directory
    base_dir = _make_path(config.get("base_dir", "../runs"), experiment_root)
    run_name = config.get("run_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{run_name}"
    output_dir = base_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Runner] Output → {output_dir}")

    # ------------------------------------------------------------------
    # 1. DATA INGESTION
    # ------------------------------------------------------------------
    print("\n>>> 1. Loading Datasets...")
    data_cfg = config["data"]

    datasets = []
    for ds_cfg in data_cfg["datasets"]:
        extra = {}
        extra["global_config"] = global_config
        ds = _instantiate_from_cfg(ds_cfg, extra_kwargs=extra, experiment_root=experiment_root)
        print(f"   Processing {ds.config['dataset'].get('name', ds_cfg['class'])}...")
        ds.clean()
        ds.standardize()
        datasets.append(ds)

    # ------------------------------------------------------------------
    # 1b. FEATURE RESOLUTION
    # ------------------------------------------------------------------
    sample_df = datasets[0].all_data[0]
    feature_cfg = config.get("features", {})

    features = resolve_features(
        df=sample_df,
        schema=schema,
        use_static=feature_cfg.get("use_static", True),
        add_masks=feature_cfg.get("add_masks", True),
        mask_suffix=feature_cfg.get("mask_suffix", "_mask"),
    )
    target_col = features["target"]
    static_cols = features["static"]
    dynamic_cols = features["dynamic"]
    all_feature_cols = features["all_features"]

    print(f"   Target: {target_col}")
    print(f"   Static features ({len(static_cols)}): {static_cols}")
    print(f"   Dynamic features ({len(dynamic_cols)}): {dynamic_cols}")
    print(f"   Total conditional features: {len(all_feature_cols)}")

    # ------------------------------------------------------------------
    # 2. SPLITTING
    # ------------------------------------------------------------------
    print("\n>>> 2. Splitting Datasets...")
    split_cfg = data_cfg.get("split", {})
    splitter_cls = registry.get(split_cfg.get("class", "DataSplitter"))
    splitter = splitter_cls(
        val_ratio=split_cfg.get("val_ratio", 0.15),
        random_state=split_cfg.get("random_state", 42),
    )
    train_dfs_raw, val_dfs_raw = splitter.split_data(
        datasets=datasets,
        strategy=split_cfg.get("strategy", "subject"),
    )

    # ------------------------------------------------------------------
    # 3. NORMALIZATION
    # ------------------------------------------------------------------
    print("\n>>> 3. Normalizing Data...")
    norm_cfg = config.get("normalization", {})
    normalizer_cls = registry.get(norm_cfg.get("class", "MinMaxNormalizer"))

    norm_params = copy.deepcopy(norm_cfg.get("params", {}))
    norm_params["cols_to_normalize"] = all_feature_cols + [target_col]

    # Backward compat: support top-level feature_range
    if "feature_range" in norm_cfg and "feature_range" not in norm_params:
        norm_params["feature_range"] = tuple(norm_cfg["feature_range"])
    if "fixed_ranges" not in norm_params:
        fixed_ranges = global_config.get("normalization_ranges", None)
        if fixed_ranges is not None:
            norm_params.setdefault("fixed_ranges", fixed_ranges)

    # Only pass kwargs the normalizer actually accepts
    sig = inspect.signature(normalizer_cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in norm_params.items() if k in valid_keys}

    normalizer = normalizer_cls(**filtered_params)
    normalizer.fit(train_dfs_raw)
    normalizer.save_params(output_dir)
    train_dfs_norm = normalizer.transform(train_dfs_raw)
    val_dfs_norm = normalizer.transform(val_dfs_raw)

    # ------------------------------------------------------------------
    # 4. WINDOWING
    # ------------------------------------------------------------------
    print("\n>>> 4. Building Windows...")
    win_cfg = config.get("windowing", {})
    win_params = copy.deepcopy(win_cfg.get("params", {}))

    # Inject feature columns
    win_params.setdefault("target_col", target_col)
    win_params.setdefault("cond_cols", all_feature_cols)
    # Only pass static_cols if the config explicitly provides them
    # (use_static=False means we want everything as dynamic)
    if feature_cfg.get("use_static", True) and static_cols:
        win_params.setdefault("static_cols", static_cols)
    if "force_device" not in win_params:
        win_params["force_device"] = device

    builder_cls = registry.get(win_cfg.get("class", "WindowBuilder"))
    builder = builder_cls(**win_params)

    train_win_cfg = win_cfg.get("train", {})
    val_win_cfg = win_cfg.get("val", {})

    train_split = builder.build_subset(
        dfs=train_dfs_norm,
        seq_len=train_win_cfg.get("seq_len", 288),
        step=train_win_cfg.get("step", 12),
        shuffle=train_win_cfg.get("shuffle", True),
        split_name="Train",
    )
    val_split = builder.build_subset(
        dfs=val_dfs_norm,
        seq_len=val_win_cfg.get("seq_len", 288),
        step=val_win_cfg.get("step", 288),
        shuffle=val_win_cfg.get("shuffle", True),
        split_name="Validation",
    )

    # Grab a fixed batch for callbacks/visualization
    try:
        fixed_vis_batch = next(iter(val_split.loader))
    except StopIteration:
        fixed_vis_batch = None
        print("   WARNING: Validation loader is empty!")

    # Determine cond_dim from the loader (needed by some models)
    sample_batch = next(iter(train_split.loader))
    cond_dim_from_loader = sample_batch[1].shape[-1] if len(sample_batch) > 1 else 0
    print(f"   Condition dim from loader: {cond_dim_from_loader}")

    # ------------------------------------------------------------------
    # 5. TRAINING
    # ------------------------------------------------------------------
    print(f"\n>>> 5. Training...")

    # Logger
    logger_cfg = config.get("logger", {})
    logger = registry.instantiate(
        logger_cfg.get("class", "WandBLogger"),
        project_name=logger_cfg.get("project_name", "experiment"),
        run_name=experiment_name,
        config=config,
        log_dir=output_dir,
    )

    # Model
    model_cfg = config["model"]
    model_params = copy.deepcopy(model_cfg.get("params", {}))

    # Auto-inject dimensions the user shouldn't have to hard-code
    auto_dims = config.get("auto_inject_dims", True)
    if auto_dims:
        # Generic cond_dim (used by DiffWave, etc.)
        model_params.setdefault("cond_dim", cond_dim_from_loader)
        # TimeGAN-specific dims
        if static_cols and feature_cfg.get("use_static", True):
            model_params.setdefault("static_dim", len(static_cols))
            model_params.setdefault("cond_dim", len(dynamic_cols))

    model = registry.instantiate(model_cfg["class"], **model_params).to(device)

    # Trainer
    trainer_cfg = config.get("trainer", {})
    trainer = registry.instantiate(
        trainer_cfg.get("class", "Trainer"),
        device=device,
        logger=logger,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        val_check_interval=trainer_cfg.get("val_check_interval", 1),
    )

    # Phases
    phases_cfg = config.get("training", {}).get("phases", [])
    if not phases_cfg:
        # Fallback: single unnamed phase
        phases_cfg = [{"max_epochs": config.get("training", {}).get("max_epochs", 100)}]

    for phase in phases_cfg:
        phase_name = phase.get("name")
        if phase_name:
            print(f"\n   [Phase: {phase_name}]")
            if hasattr(model, "set_phase"):
                model.set_phase(phase_name)

        # Override trainer-level val_check_interval per phase if specified
        if "val_check_interval" in phase:
            trainer.val_check_interval = phase["val_check_interval"]

        callbacks = _build_callbacks(
            phase.get("callbacks"),
            fixed_batch=fixed_vis_batch,
            device=device,
        )

        trainer.fit(
            model=model,
            max_epochs=phase["max_epochs"],
            train_loader=train_split.loader,
            val_loader=val_split.loader,
            callbacks=callbacks if callbacks else None,
        )

    # Save model
    model_filename = config.get("model_filename", "model.pth")
    print(f"\n   Saving model → {output_dir / model_filename}")
    torch.save(model.state_dict(), output_dir / model_filename)
    logger.close()

    # ------------------------------------------------------------------
    # 6. INFERENCE & RECONSTRUCTION
    # ------------------------------------------------------------------
    infer_cfg = config.get("inference")
    if infer_cfg is None:
        print("\n>>> 6. Inference skipped (no 'inference' section in config).")
        config_snapshot_path = output_dir / "experiment_config.yaml"
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"   Config saved → {config_snapshot_path}")
        return output_dir

    print("\n>>> 6. Generation & Reconstruction...")

    # Builder (for inference)
    inf_builder_cfg = infer_cfg.get("builder", {})
    inf_builder_params = copy.deepcopy(inf_builder_cfg.get("params", {}))
    inf_builder_params.setdefault("target_col", target_col)
    inf_builder_params.setdefault("cond_cols", all_feature_cols)
    if feature_cfg.get("use_static", True) and static_cols:
        inf_builder_params.setdefault("static_cols", static_cols)

    inf_builder = registry.instantiate(
        inf_builder_cfg.get("class", "WindowBuilder"),
        **inf_builder_params,
    )

    # Reconstructor
    recon_cfg = infer_cfg.get("reconstructor", {})
    recon_config_obj = registry.instantiate(
        "ReconstructionConfig",
        target_col=target_col,
        cond_cols=all_feature_cols,
        include_true_target=recon_cfg.get("include_true_target", True),
    )
    recon_params = copy.deepcopy(recon_cfg.get("params", {}))
    recon_params["cfg"] = recon_config_obj
    recon_params["normalizer"] = normalizer
    reconstructor = registry.instantiate(
        recon_cfg.get("class", "WindowReconstructor"),
        **recon_params,
    )

    # Orchestrator
    orch_cfg = infer_cfg.get("orchestrator", {})
    orch_params = copy.deepcopy(orch_cfg.get("params", {}))
    # Map the right kwargs depending on orchestrator type
    orch_cls_name = orch_cfg.get("class", "InferenceOrchestrator")
    if "Sequence" in orch_cls_name:
        orch_params.setdefault("builder", inf_builder)
    else:
        orch_params.setdefault("window_builder", inf_builder)
    orch_params["model"] = model
    orch_params["reconstructor"] = reconstructor
    orch_params["device"] = device
    orch_params.setdefault("verbose", True)

    orchestrator = registry.instantiate(orch_cls_name, **orch_params)

    # Generate requested splits
    for gen_cfg in infer_cfg.get("generate", []):
        split_name = gen_cfg.get("split", "val")
        prefix = gen_cfg.get("prefix", f"{split_name}_synthetic")
        label = gen_cfg.get("label", split_name.capitalize())

        dfs = val_dfs_norm if split_name == "val" else train_dfs_norm

        print(f"\n   [Generating: {label}]...")

        # Build run kwargs – some orchestrators need seq_len, others don't
        run_kwargs: dict[str, Any] = {
            "dfs": dfs,
            "output_dir": output_dir / split_name,
            "file_prefix": prefix,
            "split_name": label,
        }
        run_kwargs.update(gen_cfg.get("run_params", {}))

        orchestrator.run(**run_kwargs)

    # Save a copy of the experiment config to the output directory
    config_snapshot_path = output_dir / "experiment_config.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"   Config saved → {config_snapshot_path}")

    print(f"\n>>> Pipeline completed. Results → {output_dir}")
    return output_dir
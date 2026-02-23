def resolve_features(df, schema, use_static=True, add_masks=True, mask_suffix="_mask"):
    """
    Analyzes schema and df to resolve feature groups
    """
    target_col = schema["target_col"]

    # 1. Static cols
    static_cols = [c for c in schema.get("static_cols", []) if c in df.columns]

    # 2. Masks inclusion
    final_static = []
    if use_static:
        for c in static_cols:
            final_static.append(c)
            if add_masks:
                mask_name = c + mask_suffix
                if mask_name in df.columns:
                    final_static.append(mask_name)

    # 3. Every feature (except target)
    all_features = [c for c in df.columns if c != target_col]

    # 4. Dynamic features
    dynamic_cols = [c for c in all_features if c not in final_static]

    return {
        "target": target_col,
        "static": final_static,
        "dynamic": dynamic_cols,
        "all_features": all_features
    }
import pandas as pd
import pytest


def test_standardize_handles_chimera_features(dummy_dataset):
    """
    Verifica la gestione "Chimera":
    - Una feature presente nel dataset ('basal_rate') deve avere mask=1.
    - Una feature MANCANTE ('heart_rate') deve essere creata a 0 con mask=0.
    """
    # 1. Setup dello Scenario
    # Configuriamo il contesto globale per richiedere una feature che c'è e una che manca
    target_col = "glucose"
    existing_feat = "basal_rate"
    missing_feat = "heart_rate"  # Questo non c'è nel toy_subject_df

    # Aggiorniamo il context del dummy dataset
    dummy_dataset.context['target_col'] = target_col
    dummy_dataset.context['global_cond_cols'] = [existing_feat, missing_feat]

    # Aggiorniamo anche lo standardizer nella pipeline (poiché è stato istanziato nell'__init__)
    # Nota: In un caso reale questo verrebbe dal global_config.yaml caricato.
    dummy_dataset.standardization_pipeline[0].global_cond_cols = [existing_feat, missing_feat]

    # Salviamo i valori originali per controllo
    original_basal = dummy_dataset.all_data[0][existing_feat].copy()

    # 2. Esecuzione
    dummy_dataset.standardize()

    df = dummy_dataset.all_data[0]

    # 3. Verifiche

    # A) Feature Esistente (Basal)
    assert existing_feat in df.columns
    # I valori devono essere preservati
    pd.testing.assert_series_equal(df[existing_feat], original_basal)
    # La maschera deve essere creata ed essere tutti 1.0
    mask_col_exist = f"{existing_feat}_mask"
    assert mask_col_exist in df.columns
    assert (df[mask_col_exist] == 1.0).all()

    # B) Feature Mancante (Heart Rate)
    assert missing_feat in df.columns
    # Deve essere stata riempita con 0.0
    assert (df[missing_feat] == 0.0).all()
    # La maschera deve essere creata ed essere tutti 0.0
    mask_col_miss = f"{missing_feat}_mask"
    assert mask_col_miss in df.columns
    assert (df[mask_col_miss] == 0.0).all()


def test_standardize_enforces_strict_schema_and_order(dummy_dataset):
    """
    Verifica che le colonne non richieste vengano rimosse e
    che l'ordine finale sia [Target, Features..., Masks...].
    """
    # 1. Setup
    # Aggiungiamo una colonna "junk" che non è nel global schema
    dummy_dataset.all_data[0]["junk_column"] = 999.0

    target_col = "glucose"
    cond_cols = ["basal_rate", "carbs"]  # Ignoriamo 'bolus_total' che pure esiste nel df

    dummy_dataset.context['target_col'] = target_col
    dummy_dataset.context['global_cond_cols'] = cond_cols
    dummy_dataset.standardization_pipeline[0].global_cond_cols = cond_cols

    # 2. Esecuzione
    dummy_dataset.standardize()
    df = dummy_dataset.all_data[0]

    # 3. Verifiche

    # A) La colonna junk deve essere sparita
    assert "junk_column" not in df.columns

    # B) La colonna 'bolus_total' (che era nel df ma non in cond_cols) deve essere sparita
    assert "bolus_total" not in df.columns

    # C) Ordine Rigoroso
    # Expected: [Target, feat1, feat2, mask1, mask2]
    expected_cols = [target_col] + cond_cols + [f"{c}_mask" for c in cond_cols]

    assert list(df.columns) == expected_cols


def test_standardize_raises_if_target_missing(dummy_dataset):
    """
    Se il target manca, deve lanciare errore (non possiamo inventarci il glucosio).
    """
    # Rimuoviamo il target dal df
    dummy_dataset.all_data[0] = dummy_dataset.all_data[0].drop(columns=["glucose"])

    dummy_dataset.context['target_col'] = "glucose"

    with pytest.raises(ValueError, match="Target column 'glucose' missing"):
        dummy_dataset.standardize()
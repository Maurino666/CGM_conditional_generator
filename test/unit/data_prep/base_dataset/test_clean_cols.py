import numpy as np
import pytest


def test_clean_pipeline_coerces_numeric_fills_defaults_and_clips(dummy_dataset):
    """
    Verifica che ds.clean() (che chiama TypeAndValueCleaner) funzioni come previsto.
    """
    # 1. Sporchiamo i dati manualmente nel primo soggetto
    df = dummy_dataset.all_data[0].copy()  # Lavoriamo su copia per sicurezza

    # A) Valore stringa in colonna numerica
    df["glucose"] = df["glucose"].astype(object)
    df.loc[df.index[0], "glucose"] = "not_a_number"

    # B) NaN in colonna che ha default
    # (Assumiamo basal_rate sia nei numeric_cols e defaults)
    df.loc[df.index[1], "basal_rate"] = np.nan
    dummy_dataset.defaults["basal_rate"] = 1.23

    # C) Valori negativi nelle impulse_cols
    df.loc[df.index[2], "bolus_total"] = -5.0
    df.loc[df.index[3], "carbs"] = -10.0

    # Reinseriamo il df sporco nel dataset
    dummy_dataset.all_data[0] = df

    # 2. Eseguiamo la pulizia
    dummy_dataset.clean()

    cleaned = dummy_dataset.all_data[0]

    # 3. Asserzioni

    # A) Coercizione numerica e gestione NaN
    # "not_a_number" -> NaN (perché to_numeric con coerce)
    # Nota: TypeAndValueCleaner converte in numerico, ma NON riempie il target (glucose)
    # se non è nei defaults. Di solito glucose si gestisce con GapFiller.
    # Qui verifichiamo solo che sia diventato NaN e non crashi.
    assert np.isnan(cleaned["glucose"].iloc[0])

    # B) Riempimento Default
    assert cleaned["basal_rate"].iloc[1] == pytest.approx(1.23)

    # C) Clipping impulsi
    assert (cleaned["bolus_total"] >= 0).all()
    assert (cleaned["carbs"] >= 0).all()

    # Verifica specifica sui valori che erano negativi
    assert cleaned["bolus_total"].iloc[2] == 0.0
    assert cleaned["carbs"].iloc[3] == 0.0
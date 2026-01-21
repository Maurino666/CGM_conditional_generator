import pandas as pd
from data_prep.processors.augmentation import BaseTimeEventAugmenter


def test_augment_adds_expected_columns(dummy_dataset, toy_subject_df):
    """
    Test che verifica se BaseTimeEventAugmenter aggiunge le colonne corrette.
    Possiamo testare direttamente il processore o chiamare ds.augment().
    """
    original = toy_subject_df.copy()

    # Eseguiamo l'augmentation sull'intero dataset
    dummy_dataset.augment()

    new_df = dummy_dataset.all_data[0]

    # Verifica: le colonne originali non devono essere state rimosse/corrotte
    # (Nota: new_df avrà colonne in più, quindi controlliamo solo quelle originali)
    pd.testing.assert_frame_equal(new_df[original.columns], original)

    expected_cols = {
        "tod_sin_24h",
        "tod_cos_24h",
        "bolus_total_expdecay_120m",
        "carbs_expdecay_120m",
    }

    # Verifichiamo che le colonne attese siano presenti
    assert expected_cols.issubset(set(new_df.columns))

    # Check sui valori (le colonne decay non devono essere negative)
    assert (new_df["bolus_total_expdecay_120m"] >= 0).all()
    assert (new_df["carbs_expdecay_120m"] >= 0).all()

    # Check che qualcosa sia stato calcolato (es. dopo un bolo > 0, il decay deve essere > 0)
    assert new_df["bolus_total_expdecay_120m"].max() > 0


def test_augmentation_processor_unit_logic(dummy_dataset):
    """
    Test unitario specifico per il processore PhysiologicalDynamicsAugmenter.
    """
    processor = BaseTimeEventAugmenter()

    # Prendiamo i dati puliti
    data_list = dummy_dataset.all_data

    # Eseguiamo il processore manualmente passando il contesto del dummy dataset
    processed_list = processor.process(data_list, dummy_dataset.context)

    for df in processed_list:
        assert "tod_sin_24h" in df.columns
        assert "carbs_expdecay_120m" in df.columns
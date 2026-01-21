

def test_to_sequence_splits_conditional_subject_split(dummy_dataset):

    X_train_y, X_train_c, X_val_y, X_val_c = dummy_dataset.to_sequence_splits_conditional(
        seq_len=3,
        step=1,
        cond_cols= ["basal_rate", "carbs"],
        val_ratio=0.5,
        split_by="subject",
        random_state=42,
    )

    # Con 1 soggetto in train e 1 in val, entrambi lunghi 6,
    # ci aspettiamo 4 finestre per ciascuno.
    assert X_train_y.shape == (10, 3, 1)
    assert X_train_c.shape == (10, 3, 2)
    assert X_val_y.shape == (10, 3, 1)
    assert X_val_c.shape == (10, 3, 2)
"""The only Practice 2.2 test retained here: old-import forwarding smoke."""


def test_old_practice_2_2_imports_forward_to_canonical_package():
    from processing_own_phase import data_practice_2_2 as legacy

    assert legacy.create_train_validation_datasets.__module__.startswith(
        "practice_2_2."
    )

from processing_own_phase.analyze_practice_2_2 import classify_generalization_gap


def test_generalization_gap_classification():
    assert classify_generalization_gap(15.0) == "high_overfitting_risk"
    assert classify_generalization_gap(8.0) == "moderate_overfitting_risk"
    assert classify_generalization_gap(3.0) == "controlled_gap"
    assert (
        classify_generalization_gap(-8.0)
        == "likely_underfitting_or_augmented_train_is_harder"
    )

from src.alzheimer_research.train_baseline import run_baseline


def test_baseline_has_expected_quality_threshold():
    results_df = run_baseline(verbose=False)
    assert results_df["ROC-AUC"].max() > 0.7

from vectorizacion import evaluate_similarity_score


def test_evaluate_similarity_score_thresholds():
    assert evaluate_similarity_score(0.81)[0].startswith("🟢")
    assert evaluate_similarity_score(0.61)[0].startswith("🟢")
    assert evaluate_similarity_score(0.41)[0].startswith("🟡")
    assert evaluate_similarity_score(0.21)[0].startswith("🟡")
    assert evaluate_similarity_score(0.10)[0].startswith("🔴")

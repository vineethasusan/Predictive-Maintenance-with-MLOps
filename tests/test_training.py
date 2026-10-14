import json

from fleetpm.features import FEATURES


def test_training_produces_gated_model_and_card(trained):
    card, art = trained
    assert (art / "model_card.json").exists() or (art / "rejected_candidate.json").exists()
    if card["quality_gate_passed"]:
        assert (art / "model.joblib").exists()
        assert card["features"] == FEATURES
        m = card["test_metrics"]
        assert 0.5 < m["roc_auc"] <= 1.0
        assert json.loads((art / "model_card.json").read_text())["threshold"] == card["threshold"]


def test_predictor_scores_and_explains(trained):
    import pandas as pd

    from fleetpm.predictor import Predictor

    card, art = trained
    if not card["quality_gate_passed"]:
        return
    p = Predictor(art)
    row = pd.read_csv(art / "reference_features.csv.gz").iloc[0].to_dict()
    out = p.score_features([row, row])
    assert len(out) == 2 and 0 <= out[0]["risk_score"] <= 1
    assert out[0]["risk_band"] in {"low", "medium", "high"}

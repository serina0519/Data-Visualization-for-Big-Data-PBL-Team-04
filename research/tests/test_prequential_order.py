"""Every row is scored exactly once by a model that has not seen its week; updates only on complete cycles."""
import sys, numpy as np, pandas as pd, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import experiment
from conftest import toy


@pytest.mark.parametrize('period', [1,2,4])
def test_update_schedule_and_coverage(tmp_path, period):
    d, X = toy(weeks=15)
    for sub in ['predictions','acquisition']: (tmp_path/sub).mkdir()
    s = experiment.Setting('A2', period)
    pred, res, aud = experiment.run_one(d, X, s, 0, tmp_path)
    assert len(pred) == (d.week > 4).sum() and pred.row_id.is_unique
    a = pd.DataFrame(aud)
    assert res['updates'] == (15-4)//period            # incomplete tail never triggers an update
    assert ((a.week-4) % period == 0).all()
    assert (a.train_latest_week <= a.week).all()       # never trained on future weeks


def test_scores_precede_update(tmp_path):
    """Scores for week w must equal the frozen model's output before the week-w update."""
    d, X = toy(weeks=8)
    for sub in ['predictions','acquisition']: (tmp_path/sub).mkdir()
    pred, res, aud = experiment.run_one(d, X, experiment.Setting('A2', 1), 0, tmp_path)
    # Re-fit the initial model and check week-5 scores are reproduced exactly.
    from src.models import LogisticDetector
    ini = np.flatnonzero(d.week.values <= 4)
    m = LogisticDetector(X.shape[1]).fit(X[ini], d.label.values[ini])
    w5 = pred[pred.week == 5]
    assert np.allclose(m.predict(X[w5.row_id.values]), w5.score.values)

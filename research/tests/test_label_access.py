"""Unsupervised / non-adaptive methods must never reveal post-initial truth."""
import sys, numpy as np, pandas as pd, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import experiment
from conftest import toy


@pytest.mark.parametrize('method', ['A1','B1','B2','B3','B4'])
def test_no_leak_for_unsupervised(tmp_path, method):
    d, X = toy()
    for sub in ['predictions','acquisition']: (tmp_path/sub).mkdir()
    s = experiment.Setting(method, 0 if method=='A1' else 1)
    pred, res, aud = experiment.run_one(d, X, s, 0, tmp_path)
    assert res['post_initial_labels'] == 0
    led = pd.read_csv(tmp_path/'acquisition'/f'{s.name}_s0.csv.gz')
    assert (led.reveal_week <= 4).all()


def test_oracle_blocks_unsupervised_leak():
    o = experiment.LabelOracle(np.array([0,1,0,1]), [0,1], 'B1')
    with pytest.raises(AssertionError):
        o.reveal([2], 5, 'bad')


def test_budget_methods_reveal_only_selected(tmp_path):
    d, X = toy()
    for sub in ['predictions','acquisition']: (tmp_path/sub).mkdir()
    for method in ['C1','C2','R1']:
        s = experiment.Setting(method, 1, .10, 'V2' if method=='R1' else 'V1')
        pred, res, aud = experiment.run_one(d, X, s, 0, tmp_path)
        a = pd.DataFrame(aud)
        assert (a.truth_new == np.floor(.10*a.pool_n)).all()   # exact floor budget, no min-1 rule
        assert res['post_initial_labels'] == a.truth_new.sum()

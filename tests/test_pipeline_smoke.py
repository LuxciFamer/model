import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.dataset import FeatureScaler, monthly_index  # noqa: E402
from src.data.meteo_et0 import fao56_et0  # noqa: E402
from src.evaluation.metrics import all_metrics, kge, nse  # noqa: E402
from src.models.losses import composite_loss  # noqa: E402
from src.models.pinn_hydro import MetaPINNHydro, NeuralOnly, train_seq_model  # noqa: E402
from tests.test_physics_conservation import BOUNDS, _forcing  # noqa: E402

PRIOR = {"Tth": 0.5, "Tm": 1.0, "Df": 2.5, "cn_scale": 1.0, "lam_ia": 0.15, "Smax": 180.0,
         "Qmax": 0.4, "f": 0.03, "alpha_et": 0.75, "kf": 0.45, "ks": 0.02}


def test_metrics_perfect_and_reference():
    o = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(nse(o, o) - 1) < 1e-12 and abs(kge(o, o)[0] - 1) < 1e-12
    assert abs(nse(o, np.full(5, o.mean()))) < 1e-12
    m = all_metrics(o, o * 1.1)
    assert abs(m["PBIAS"] - 10) < 1e-9


def test_fao56_reference_example():
    # FAO-56 Example 18 (Brussels, 6 July): ET0 ~ 3.9 mm/d
    et0, _ = fao56_et0(np.array(21.5), np.array(12.3), np.array(22.07), np.array(2.078),
                       np.array(((63 + 84) / 2) / 100), np.array(50.8), np.array(100.0), np.array(187))
    assert 3.3 < float(et0) < 4.4


def test_scaler_fits_only_calibration():
    X = np.arange(20, dtype=float).reshape(10, 2)
    m = np.zeros(10, bool); m[:4] = True
    s = FeatureScaler().fit(X, m)
    assert np.allclose(s.mean_, X[:4].mean(0))


def test_hybrid_forward_backward_and_training():
    P, T, E = _forcing(500)
    X = torch.randn(500, 5)
    model = MetaPINNHydro(5, BOUNDS, 75.0, PRIOR, hidden=8)
    Pt, Tt, Et = map(torch.tensor, (P, T, E))
    out = model(X, Pt, Tt, Et)
    assert out["Q"].shape == (500,) and torch.isfinite(out["Q"]).all() and (out["Q"] >= 0).all()
    q = out["Q"].detach().numpy() * 1.2 + 0.01
    mask = torch.zeros(500, dtype=torch.bool); mask[100:400] = True
    loss, terms = composite_loss(out, torch.tensor(q), mask, {"lambda_wb": 1, "lambda_ode": 1,
                                                              "lambda_mono": 1, "lambda_res": 1}, float(q.mean()))
    loss.backward()
    assert set(terms) == {"data", "wb", "ode", "res", "mono"}
    m_cal = np.zeros(500, bool); m_cal[100:400] = True
    m_val = np.zeros(500, bool); m_val[400:] = True
    model2 = MetaPINNHydro(5, BOUNDS, 75.0, PRIOR, hidden=8)
    _, hist, best = train_seq_model(model2, X, Pt, Tt, Et, {"daily": q}, m_cal, m_val,
                                    {"lambda_wb": 0.1}, 0.01, 3, 5)
    assert len(hist) == 3 and np.isfinite(best)


def test_monthly_loss_interface():
    idx = pd.date_range("2015-01-01", periods=400, freq="D")
    mi = monthly_index(idx)
    P, T, E = _forcing(400)
    model = NeuralOnly(4, hidden=8)
    out = model(torch.randn(400, 4), torch.tensor(P), torch.tensor(T), torch.tensor(E))
    qm = torch.rand(int(mi.max()) + 1, dtype=torch.float64)
    mm = torch.zeros(len(qm), dtype=torch.bool); mm[1:10] = True
    mask = torch.ones(400, dtype=torch.bool)
    loss, _ = composite_loss(out, None, mask, {"lambda_mono": 0.1}, 1.0,
                             month=(torch.as_tensor(mi), qm, mm))
    loss.backward()
    assert torch.isfinite(loss)

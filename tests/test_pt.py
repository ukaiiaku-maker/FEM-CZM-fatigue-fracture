import numpy as np
from fem_czm.parameters import get_material
from fem_czm.peierls_taylor import evaluate_pt_rates


def test_detailed_balance_zero_stress_gives_zero_net_flow():
    m = get_material("DBTT")
    r = evaluate_pt_rates(0.0, 1e14, 1e12, 700.0, 2.74e-10, m)
    assert float(r.peierls_net_s) == 0.0
    assert float(r.taylor_net_s) == 0.0
    assert float(r.equivalent_plastic_rate_s) == 0.0


def test_taylor_hit_order_is_uncapped_and_density_dependent():
    m = get_material("ceramic")
    rho = np.array([1e10, 1e14, 1e18])
    r = evaluate_pt_rates(np.full(3, 3e9), rho, np.full(3, 1e12), 700.0, 2.74e-10, m)
    assert np.all(np.diff(r.hit_order) > 0.0)
    assert r.hit_order[-1] > r.hit_order[0]

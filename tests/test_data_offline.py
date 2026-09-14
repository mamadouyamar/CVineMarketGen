"""Offline checks of the data module: constants and cache handling, no download."""
import pandas as pd
from cvinemarketgen import data


def test_factors_has_small_cap():
    assert data.FACTORS[-1] == 'Small cap' and len(data.FACTORS) == 7


def test_load_factor_data_refreshes_old_cache(tmp_path, monkeypatch):
    old = pd.DataFrame({c: [0.01, 0.02] for c in data.FACTORS[:-1]},
                       index=pd.PeriodIndex(['2020-01', '2020-02'], freq='M'))
    p = tmp_path / 'factors_cache.csv'
    old.to_csv(p)
    called = {}

    def fake_download(start, end, cache, D, verbose):
        called['yes'] = True
        return old.assign(**{'Small cap': [0.0, 0.0]})
    monkeypatch.setattr(data, '_download_factors', fake_download)
    df = data.load_factor_data(cache=str(p), verbose=False)
    assert called and list(df.columns) == data.FACTORS


def test_load_etf_monthly_from_cache(tmp_path):
    idx = pd.PeriodIndex(['2020-01', '2020-02', '2020-03'], freq='M')
    pd.DataFrame({'SPY': [0.01, -0.02, 0.03], 'TLT': [0.0, 0.01, -0.01]}, index=idx).to_csv(tmp_path / 'etf.csv')
    df = data.load_etf_monthly(['TLT', 'SPY'], cache=str(tmp_path / 'etf.csv'), verbose=False)
    assert list(df.columns) == ['TLT', 'SPY'] and isinstance(df.index, pd.PeriodIndex) and df.shape == (3, 2)

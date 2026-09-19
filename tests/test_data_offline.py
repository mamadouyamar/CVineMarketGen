"""Offline checks of the data module: constants and cache handling, no download."""
import numpy as np
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


def test_load_fred_monthly_offline(monkeypatch, tmp_path):
    import cvinemarketgen.data as data
    idx = pd.date_range('2020-01-01', '2020-03-31', freq='B')
    fake = {'DGS2': pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx, name='DGS2'),
            'DGS10': pd.Series(np.linspace(2.0, 3.0, len(idx)), index=idx, name='DGS10')}
    monkeypatch.setattr(data, 'fred_series', lambda sid, start='1986-01-01', timeout=60: fake[sid].loc[start:])
    cache = tmp_path / 'fred.csv'
    df = data.load_fred_monthly(['DGS2', 'DGS10'], start='2020-01', cache=str(cache), verbose=False)
    assert list(df.columns) == ['DGS2', 'DGS10'] and df.index.freqstr == 'M' and len(df) == 3
    assert abs(df.loc['2020-02', 'DGS2'] - fake['DGS2'].loc['2020-02'].mean()) < 1e-12
    monkeypatch.setattr(data, 'fred_series', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('network')))
    df2 = data.load_fred_monthly(['DGS2'], start='2020-01', cache=str(cache), verbose=False)   # served from the cache
    assert df2.shape == (3, 1)


def test_load_fred_monthly_cache_stale_start_and_merge(monkeypatch, tmp_path):
    idx = pd.date_range('2010-01-01', '2021-12-31', freq='D')
    fake = {sid: pd.Series(np.linspace(1, 2, len(idx)) + k, index=idx) for k, sid in enumerate(['A', 'B', 'C'])}
    monkeypatch.setattr(data, 'fred_series', lambda sid, start='1986-01-01', timeout=60: fake[sid].loc[start:])
    cache = tmp_path / 'fred.csv'
    df1 = data.load_fred_monthly(['A', 'B'], start='2015-01', cache=str(cache), verbose=False)
    assert str(df1.index.min()) == '2015-01'
    df2 = data.load_fred_monthly(['A'], start='2011-01', cache=str(cache), verbose=False)   # cache starts later: refetch
    assert str(df2.index.min()) == '2011-01'
    cached = pd.read_csv(cache, index_col=0)
    assert set(cached.columns) == {'A', 'B'}                                                # B kept in the merged cache
    df3 = data.load_fred_monthly(['C'], start='2012-01', cache=str(cache), verbose=False)
    assert set(pd.read_csv(cache, index_col=0).columns) == {'A', 'B', 'C'} and str(df3.index.min()) == '2012-01'
    monkeypatch.setattr(data, 'fred_series', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('network')))
    df4 = data.load_fred_monthly(['A', 'B'], start='2016-01', cache=str(cache), verbose=False)   # served from the cache
    assert str(df4.index.min()) == '2016-01' and list(df4.columns) == ['A', 'B']

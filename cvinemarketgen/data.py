# -*- coding: utf-8 -*-
"""
Point-in-time market factor data for the macro-factor example.

Nothing is shipped with the package: series are downloaded at run time from
their public endpoints and cached locally as a CSV. All factors are monthly
excess returns in decimal:

  Equity DM    developed-market equity excess return             Fama-French, Developed 3 factors, Mkt-RF
  Equity EM    emerging minus developed market excess return     Fama-French, Emerging 5 factors minus Developed
  Inflation    long TIPS / short nominal Treasury                FRED T10YIE (10y breakeven): carry + D * change
  Real premia  long TIPS / short cash                            FRED DFII10 (10y TIPS real yield): carry - D * change
  Credit       long Baa corporates / short Treasuries            FRED BAA10Y (Moody's Baa minus 10y Treasury): carry - D * change
  Commodity    broad commodity futures ETF minus cash            Yahoo Finance DBC adjusted close minus Fama-French RF

The three yield-based factors are duration-scaled proxies of the excess return
of the corresponding long/short position (carry from the level, price return
from the yield change), the Duration-Times-Spread decomposition of Ben Dor et
al. (2007) for the credit leg. Durations are constants, see DURATIONS.

Sources and terms: Fama-French data library (free for research, cite the
library); FRED (Treasury-derived series are public domain; Moody's series is a
third-party series on FRED, download and use for research, do not redistribute
the file); Yahoo Finance (research use, downloaded at run time).
"""
import io
import os
import json
import zipfile

import numpy as np
import pandas as pd
import requests

UA = {'User-Agent': 'CVineMarketGen/0.1 (research; contact mamadou-yamar.thioub@hec.ca)'}
DURATIONS = {'Inflation': 8.0, 'Real premia': 8.0, 'Credit': 10.0}
FACTORS = ['Equity DM', 'Equity EM', 'Real premia', 'Inflation', 'Credit', 'Commodity']


# ----------------------------------------------------------------------------
# raw downloads
# ----------------------------------------------------------------------------
def fred_series(series_id, start='1986-01-01', timeout=60):
    """Daily FRED series as a pandas Series indexed by date (NaN for missing)."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}'
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values='.')
    df.columns = ['DATE', series_id]
    s = pd.Series(df[series_id].values, index=pd.to_datetime(df['DATE']), name=series_id)
    return s.dropna()


def fama_french_monthly(name, timeout=60):
    """Monthly table of a Fama-French CSV zip (values in percent), PeriodIndex('M')."""
    url = f'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip'
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    lines = z.read(z.namelist()[0]).decode('latin1').replace('\r', '').split('\n')
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(',Mkt-RF'))
    header = [h.strip() for h in lines[start].split(',')][1:]
    rows = []
    for l in lines[start + 1:]:
        p = [x.strip() for x in l.split(',')]
        if len(p[0]) != 6 or not p[0].isdigit():
            break
        rows.append([p[0]] + [float(x) for x in p[1:]])
    out = pd.DataFrame(rows, columns=['ym'] + header).set_index('ym')
    out.index = pd.PeriodIndex(out.index, freq='M')
    return out


def yahoo_monthly_adjclose(ticker, timeout=60):
    """Monthly adjusted close (dividends reinvested) from Yahoo Finance, PeriodIndex('M')."""
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=max&interval=1mo'
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    res = json.loads(r.text)['chart']['result'][0]
    ts = pd.to_datetime(res['timestamp'], unit='s')
    adj = res['indicators']['adjclose'][0]['adjclose']
    s = pd.Series(adj, index=ts, name=ticker).dropna()
    s.index = s.index.to_period('M')
    return s[~s.index.duplicated(keep='last')]


# ----------------------------------------------------------------------------
# factor construction
# ----------------------------------------------------------------------------
def _yield_factor(daily_pct, duration, sign):
    """
    Monthly excess return of a long/short position from a daily yield or spread
    in percent: carry (level / 12) plus sign * duration * monthly change.
    sign = -1 for a long bond position (price falls when the yield rises),
    sign = +1 for a long TIPS / short nominal position on the breakeven.
    """
    m = daily_pct.groupby(daily_pct.index.to_period('M')).last() / 100.0   # month-end value, PeriodIndex('M')
    return (m.shift(1) / 12.0 + sign * duration * m.diff()).dropna()


def load_factor_data(start='2006-03', end=None, cache='data/factors_cache.csv',
                     durations=None, refresh=False, verbose=True):
    """
    Monthly excess returns (decimal) of the six market factors, PeriodIndex('M'),
    columns in the order of FACTORS. Downloaded from Fama-French, FRED and Yahoo
    Finance, then cached to `cache` (set refresh=True to download again).
    """
    if cache and os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq='M')
        if verbose:
            print(f'factor data read from cache {cache}: {df.shape[0]} months, {df.index.min()} to {df.index.max()}')
        return df.loc[start:end] if end else df.loc[start:]

    D = dict(DURATIONS)
    if durations:
        D.update(durations)

    dm = fama_french_monthly('Developed_3_Factors')
    em = fama_french_monthly('Emerging_5_Factors')
    rf = dm['RF'] / 100.0
    equity_dm = dm['Mkt-RF'] / 100.0
    equity_em = (em['Mkt-RF'] - dm['Mkt-RF']) / 100.0

    breakeven = fred_series('T10YIE', start='2003-01-01')
    real_yield = fred_series('DFII10', start='2003-01-01')
    baa_spread = fred_series('BAA10Y', start='1986-01-01')
    inflation = _yield_factor(breakeven, D['Inflation'], sign=+1)
    real_premia = _yield_factor(real_yield, D['Real premia'], sign=-1)
    credit = _yield_factor(baa_spread, D['Credit'], sign=-1)

    dbc = yahoo_monthly_adjclose('DBC')
    commodity = (dbc.pct_change() - rf).dropna()

    df = pd.concat({'Equity DM': equity_dm, 'Equity EM': equity_em, 'Real premia': real_premia,
                    'Inflation': inflation, 'Credit': credit, 'Commodity': commodity}, axis=1)
    df = df[FACTORS].dropna()
    df = df.loc[start:end] if end else df.loc[start:]
    df.index.name = 'month'
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        df.to_csv(cache)
    if verbose:
        print(f'factor data downloaded: {df.shape[0]} months, {df.index.min()} to {df.index.max()}'
              + (f', cached to {cache}' if cache else ''))
    return df


def factor_targets(returns):
    """
    Targets read from the sample itself, in the layout expected by
    CVineGenerator.load_data_and_setup: DataFrame indexed by factor with
    'Arithmetic Mean', 'Volatility' and the correlation-matrix columns.
    Monthly units, as the returns.
    """
    cols = list(returns.columns)
    out = pd.DataFrame(index=pd.Index(cols, name='Assets'))
    out['Arithmetic Mean'] = returns.mean().values
    out['Volatility'] = returns.std(ddof=1).values
    corr = returns.corr()
    for c in cols:
        out[c] = corr[c].values
    return out


# ----------------------------------------------------------------------------
# daily prices, for the backtesting example
# ----------------------------------------------------------------------------
def yahoo_daily_adjclose(ticker, start=None, timeout=60):
    """Daily adjusted close (dividends reinvested) from Yahoo Finance, DatetimeIndex."""
    p1 = int(pd.Timestamp(start or '1990-01-01').timestamp())
    p2 = int(pd.Timestamp.utcnow().timestamp()) + 86400
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={p1}&period2={p2}&interval=1d'
    r = requests.get(url, timeout=timeout, headers=UA)
    r.raise_for_status()
    res = json.loads(r.text)['chart']['result'][0]
    ts = pd.to_datetime(res['timestamp'], unit='s').normalize()
    adj = res['indicators']['adjclose'][0]['adjclose']
    s = pd.Series(adj, index=ts, name=ticker).dropna()
    s = s[~s.index.duplicated(keep='last')]
    return s.loc[start:] if start else s


def load_daily_returns(tickers, start='2010-01-01', cache='data/daily_cache.csv', refresh=False, verbose=True):
    """
    Daily simple returns of a list of Yahoo Finance tickers (adjusted closes),
    aligned on common dates, cached locally as a CSV (git-ignored).
    """
    tickers = list(tickers)
    if cache and os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers) <= set(df.columns):
            df = df.loc[start:, tickers].dropna()
            if verbose:
                print(f'daily returns read from cache {cache}: {df.shape[0]} days, {df.index.min().date()} to {df.index.max().date()}')
            return df
    prices = pd.concat([yahoo_daily_adjclose(t, start=start) for t in tickers], axis=1).dropna()
    df = prices.pct_change().dropna()
    df.index.name = 'date'
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        df.to_csv(cache)
    if verbose:
        print(f'daily returns downloaded: {df.shape[0]} days, {df.index.min().date()} to {df.index.max().date()}'
              + (f', cached to {cache}' if cache else ''))
    return df

# -*- coding: utf-8 -*-
"""
A :class:`~cvinemarketgen.simulator.MarketSpec` drawn as the file it is: column
letters, the sheet's own row numbers, a header row, empty cells left empty.

An input an analyst types in Excel is shown as an Excel sheet, and a sheet that
was just edited is shown as it now stands, the rows written shaded:

>>> show_workbook(spec)                                     # the six sheets
>>> show_change(spec2, 'assets', {'ticker': 'SPY', 'mean': 0.07})   # after an edit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .simulator import SHEET_KEY

_SHEETS = ('assets', 'tags', 'exposures', 'pairs', 'factors', 'factor_corr')


def _flat(df):
    """A sheet with its key as a column, the way the file holds it."""
    return df.reset_index() if df.index.name else df


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    return f'{v:g}' if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def draw_sheet(ax, df, title, max_rows=16, highlight=()):
    """
    One sheet into ``ax``: column letters, the row numbers of the file, a header row,
    empty cells empty, and the rows whose index label is in ``highlight`` shaded.
    """
    d = _flat(df).head(max_rows)
    pos = [int(i) + 2 if isinstance(i, (int, np.integer)) else k + 2 for k, i in enumerate(d.index)]   # the row number in the file, so a filtered view stays honest
    body = [[str(pos[k])] + [_fmt(v) for v in row] for k, row in enumerate(d.values)]
    header = ['1'] + [str(c) for c in d.columns]
    letters = [''] + [chr(65 + j) for j in range(len(d.columns))]
    widths = [max([len(str(c)) + 2] + [len(r[j + 1]) + 2 for r in body]) for j, c in enumerate(d.columns)]   # each column sized to its content
    total = sum(widths)
    cw = [0.045] + [0.955 * w / total for w in widths]
    hi = {k for k, i in enumerate(d.index) if i in set(highlight)}
    ax.axis('off')
    tab = ax.table(cellText=[letters, header] + body, loc='upper left', cellLoc='left', colWidths=cw)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.3)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#c8c8c8')
        if r == 0 or c == 0:
            cell.set_facecolor('#e8e8e8'); cell.set_text_props(color='#555555', ha='center')
        elif r == 1:
            cell.set_facecolor('#dbe5f1'); cell.set_text_props(weight='bold')
        elif r - 2 in hi:
            cell.set_facecolor('#fff2cc')
    ax.set_title(title, loc='left', fontsize=10, pad=2)
    return total


def show_sheet(df, title, max_rows=16, highlight=()):
    """One sheet in its own figure."""
    d = _flat(df).head(max_rows)
    w = sum(max([len(str(c)) + 2] + [len(_fmt(v)) + 2 for v in d[c]]) for c in d.columns)
    fig, ax = plt.subplots(figsize=(min(0.11 * (w + 6), 16), 0.32 * (len(d) + 2) + 0.4))
    draw_sheet(ax, df, title, max_rows, highlight)
    plt.show()


def sheet_view(df, keep=(), max_rows=14):
    """A long sheet cut to the rows worth reading, in file order, their row numbers kept."""
    d = _flat(df)
    if len(d) <= max_rows or not len(keep):
        return d.head(max_rows)
    return d[d[d.columns[0]].astype(str).isin([str(k) for k in keep])]


def show_change(spec, name, added, keep=(), title=None, max_rows=14):
    """
    The sheet ``name`` of ``spec`` as it now stands: the rows already in the file for
    context, the rows of ``added`` (a dict, a list of dicts or a DataFrame, keyed as
    ``add``) shaded, and the rows named in ``keep`` held whatever happens.
    """
    if isinstance(added, dict):
        added = [added]
    d = _flat(getattr(spec, name))
    k = SHEET_KEY[name]
    tup = set(map(tuple, pd.DataFrame(added).reindex(columns=k).astype(str).values))
    hi = [i for i, row in zip(d.index, d[k].astype(str).values) if tuple(row) in tup]
    if len(d) > max_rows:
        wanted = set(hi) | set(d.index[d[k[0]].astype(str).isin([str(x) for x in keep])])
        d = d.loc[sorted(wanted)]
    show_sheet(d, title or f'sheet {name}, as it now stands ({len(_flat(getattr(spec, name)))} rows in the file)',
               max_rows=max_rows, highlight=hi)


def informative(df, max_rows):
    """The rows of a sheet that carry the most, in file order, their row numbers kept."""
    d = _flat(df)
    if len(d) <= max_rows:
        return d
    filled = d.iloc[:, 1:].notna().sum(axis=1)
    return d.loc[sorted(filled.sort_values(ascending=False, kind='stable').head(max_rows).index)]


def show_workbook(spec, max_rows=4):
    """The six sheets of a workbook, stacked, as you find them when you open the file."""
    sheets = [(n, informative(getattr(spec, n), max_rows)) for n in _SHEETS]
    rows = [min(len(d), max_rows) + 2 for _, d in sheets]          # panel heights follow the rows each sheet shows
    fig, axes = plt.subplots(len(sheets), 1, figsize=(11, 0.34 * sum(rows) + 0.3 * len(sheets)),
                             gridspec_kw={'height_ratios': rows})
    for ax, (name, d) in zip(axes, sheets):
        draw_sheet(ax, d, name, max_rows=max_rows)
    plt.tight_layout(h_pad=0.9)
    plt.show()

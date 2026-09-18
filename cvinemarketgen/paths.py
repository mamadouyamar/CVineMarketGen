# -*- coding: utf-8 -*-
"""Container for simulated return paths."""
import numpy as np
import pandas as pd


class Paths:
    """
    Simulated return paths: an array of shape ``(n_paths, horizon, N)`` with asset names.

    Attributes
    ----------
    array : numpy.ndarray
        ``(n_paths, horizon, N)`` returns per period.
    assets : list of str
    """

    def __init__(self, array, assets, layer='returns'):
        self.array = np.asarray(array, float)
        self.assets = list(assets)
        self.layer = layer
        if self.array.ndim != 3 or self.array.shape[2] != len(self.assets):
            raise ValueError('array must have shape (n_paths, horizon, N)')

    @property
    def n_paths(self):
        return self.array.shape[0]

    @property
    def horizon(self):
        return self.array.shape[1]

    def to_frame(self):
        """Long DataFrame with columns ``path``, ``t`` and one column per asset."""
        n, h, N = self.array.shape
        idx = pd.MultiIndex.from_product([range(n), range(1, h + 1)], names=['path', 't'])
        return pd.DataFrame(self.array.reshape(n * h, N), index=idx, columns=self.assets)

    def wide(self, asset):
        """DataFrame ``horizon x n_paths`` of one asset, one column per path."""
        j = self.assets.index(asset)
        return pd.DataFrame(self.array[:, :, j].T, index=range(1, self.horizon + 1),
                            columns=[f'path {i}' for i in range(self.n_paths)])

    def cumulative(self):
        """Paths of cumulative returns ``prod(1 + r) - 1`` along the horizon (meaningful for return columns only)."""
        return Paths(np.cumprod(1.0 + self.array, axis=1) - 1.0, self.assets, layer='cumulative returns')

    def terminal(self):
        """DataFrame ``n_paths x N`` of cumulative returns at the end of the horizon (meaningful for return columns only)."""
        return pd.DataFrame(np.prod(1.0 + self.array, axis=1) - 1.0, columns=self.assets)

    def summary(self):
        """Per-asset mean, volatility, skewness and kurtosis of the per-period returns pooled over paths."""
        flat = pd.DataFrame(self.array.reshape(-1, len(self.assets)), columns=self.assets)
        return pd.concat([flat.mean().rename('mean'), flat.std(ddof=1).rename('vol'),
                          flat.skew().rename('skew'), (flat.kurtosis() + 3).rename('kurt')], axis=1)

    def __repr__(self):
        return f'Paths(n_paths={self.n_paths}, horizon={self.horizon}, assets={len(self.assets)}, {self.layer})'

import numpy as np
import pandas as pd

from cvinemarketgen.paths import Paths, price_level, deflate, yoy


def test_price_level_and_yoy():
    pi = np.full((3, 24, 1), 2.4)                      # 2.4 percent a year, every month
    P = Paths(pi, ['pi'], layer='mixed')
    L = price_level(P, 'pi')
    assert L.shape == (3, 24)
    assert np.allclose(L[:, 11], 100 * np.exp(0.024)) and np.allclose(L[:, 23], 100 * np.exp(0.048))
    hist = pd.Series(np.full(30, 2.4))
    Y = yoy(P, 'pi', hist)
    assert Y.shape == (3, 24) and np.allclose(Y, 100 * (np.exp(0.024) - 1))
    Y2 = yoy(P, 'pi', pd.Series(np.zeros(30)))
    assert np.isclose(Y2[0, 0], 100 * (np.exp(0.002) - 1)) and np.isclose(Y2[0, 11], 100 * (np.exp(0.024) - 1))
    assert np.isclose(Y2[0, 23], 100 * (np.exp(0.024) - 1))


def test_deflate():
    A = np.zeros((2, 6, 2)); A[:, :, 0] = 0.01; A[:, :, 1] = 12.0   # 1 percent a month nominal, 1 percent a month inflation
    P = Paths(A, ['SPY', 'pi'], layer='mixed')
    R = deflate(P, 'SPY', 'pi')
    assert isinstance(R, Paths) and R.assets == ['SPY (real)'] and R.array.shape == (2, 6, 1)
    assert np.allclose(R.array, 1.01 / np.exp(0.01) - 1)

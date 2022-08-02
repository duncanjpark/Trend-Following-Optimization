from backtesting.test import SMA
from backtesting.lib import SignalStrategy
import pandas as pd
import numpy as np

class SmaCross(SignalStrategy):
    n1 = 10
    n2 = 25

    def init(self):
        super.init()

        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)

        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        signal = signal.replace(-1, 0)

        entry_size = signal * 0.95

        self.set_signal(entry_size=entry_size)

        
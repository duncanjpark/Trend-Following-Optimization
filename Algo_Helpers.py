import pandas as pd
from IPython.display import display
import bt


class Signal(bt.Algo):


    def __init__(self, lookback=pd.DateOffset(months=3),
                 lag=pd.DateOffset(days=0)):
        super(Signal, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        """
        selected = 'aapl'
        t0 = target.now - self.lag

        if target.universe[selected].index[0] > t0:
            return False
        prc = target.universe[selected].loc[t0 - self.lookback:t0]


        trend = prc.iloc[-1]/prc.iloc[0] - 1
        signal = trend > 0.

        if signal:
            target.temp['Signal'] = 1
        else:
            target.temp['Signal'] = 0
        """

        for selected in target.perm['tickers']:
            #display("sig: " + selected)

            t0 = target.now - self.lag

            if target.universe[selected].index[0] > t0:
                return False
            prc = target.universe[selected].loc[t0 - self.lookback:t0]

            trend = prc.iloc[-1]/prc.iloc[0] - 1
            signal = trend > 0.

            if signal:
                target.temp[selected] = 1
            else:
                target.temp[selected] = 0

        return True



class WeighFromSignal(bt.Algo):


    def __init__(self):
        super(WeighFromSignal, self).__init__()

    def __call__(self, target):
        """selected = 'aapl'
        if target.temp['Signal'] is None:
            raise(Exception('No Signal!'))

        target.temp['weights'] = {selected : target.temp['Signal']}
        return True"""
        #display(target.perm['tickers'])
        target.temp['weights'] = {'aapl': target.temp['aapl']}
        for selected in target.perm['tickers']:
            if target.temp[selected] is None:
                raise(Exception('No Signal!'))

            target.temp['weights'][selected] = target.temp[selected]


        display(target.temp['weights'])
        return True

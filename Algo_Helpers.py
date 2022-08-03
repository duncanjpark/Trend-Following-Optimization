import pandas as pd
from IPython.display import display
import bt
import numpy as np

import matplotlib.pyplot as plt



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
        #display("here")

        target.temp['Signal'] = {}

        for selected in target.perm['tickers']:

            t0 = target.now - self.lag

            if target.universe[selected].index[0] > t0:
                return False
            prc = target.universe[selected].loc[t0 - self.lookback:t0]

            trend = prc.iloc[-1]/prc.iloc[0] - 1
            signal = trend > 0.

            #display("sig: " + selected)

            #target.temp['Signal'] = {selected: 0}

            if signal:
                target.temp['Signal'][selected] = .2
            else:
                target.temp['Signal'][selected] = 0

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
        #display(target.temp.items())

        #target.temp['weights'] = {'aapl': target.temp['Signal']['aapl']}
        target.temp['weights'] = {}
        for selected in target.perm['tickers']:
            if target.temp['Signal'][selected] is None:
                raise(Exception('No Signal!'))

            target.temp['weights'][selected] = target.temp['Signal'][selected]

        #target.temp['cash'] = 0.4
        return True

class Rebalance(bt.Algo):

    def __init__(self):
        super(Rebalance, self).__init__()

    def __call__(self, target):


        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        # save value because it will change after each call to allocate
        # use it as base in rebalance calls
        # call it before de-allocation so that notional_value is correct
        if target.fixed_income:
            if "notional_value" in target.temp:
                base = target.temp["notional_value"]
            else:
                base = target.notional_value
        else:
            base = target.value

        # de-allocate children that are not in targets and have non-zero value
        # (open positions)
        for cname in target.children:
            # if this child is in our targets, we don't want to close it out
            if cname in targets:
                continue

            # get child and value
            c = target.children[cname]
            if target.fixed_income:
                v = c.notional_value
            else:
                v = c.value

            # if non-zero and non-null, we need to close it out
            if v != 0.0 and not np.isnan(v):
                target.close(cname, update=False)

        # If cash is set (it should be a value between 0-1 representing the
        # proportion of cash to keep), calculate the new 'base'
        if "cash" in target.temp and not target.fixed_income:
            base = base * (1 - target.temp["cash"])

        # Turn off updating while we rebalance each child
        

        for item in targets.items():
            #display(base)
            target.rebalance(item[1], child=item[0], base=base, update=False)

        # Now update
        target.root.update(target.now)

        return True
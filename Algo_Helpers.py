import pandas as pd
from IPython.display import display
import bt
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt


def solve_weights(rdf):
    Sigma = np.cov(rdf.transpose())
    n = Sigma.shape[0]
    mu = rdf.mean().values
    w = cp.Variable(n)
    ret = mu.T @ w
    vol = cp.quad_form(w, Sigma)
    
    prob = cp.Problem(cp.Minimize(vol),
        [cp.sum(w) == 1,
        w >= 0]
        )

    prob.solve()
    minvol = vol.value

    prob = cp.Problem(cp.Maximize(ret),  # maximize return
            [cp.sum(w) == 1, 
            w >= 0]
            )

    prob.solve()
    maxretvol = vol.value

    max_vol = (maxretvol + minvol) / 2 
    prob = cp.Problem(cp.Maximize(ret),
                  [cp.sum(w) == 1, 
                   w >= 0,
                   vol <= max_vol   # new constraint: vol <= vol_limit parameter
                  ]
                 )

    result = prob.solve()
    return w.value


class Signal(bt.Algo):


    def __init__(self, lookback_months=3,
                 lag_days=1):
        super(Signal, self).__init__()
        self.lookback_months = lookback_months
        self.lag_days = lag_days
        self.lookback = pd.DateOffset(months=lookback_months)
        self.lag = pd.DateOffset(days=lag_days)

    def __call__(self, target):
        target.temp['Signal'] = {}
        t0 = target.now - self.lag
        """for selected in target.perm['tickers']:

            if target.universe[selected].index[0] > t0:
                return False
            prc = target.universe[selected].loc[t0 - self.lookback:t0]

            sma  = prc.rolling(window=21*12,center=False).median().shift(self.lag_days)

            trend = prc.iloc[-1]/prc.iloc[0] - 1
            signal = trend > 0.

            if signal:
                target.temp['Signal'][selected] = 1
            else:
                target.temp['Signal'][selected] = 0
"""
        
        prc = target.universe.loc[t0 - self.lookback:t0, :]

        #sma  = prc.rolling(window=21*self.lookback_months,center=False).median().shift(self.lag_days)
        sma  = prc.rolling(window=21*1,center=False).median().shift(self.lag_days)
        
        for selected in target.perm['tickers']:
            target.temp['Signal'][selected] = prc.iloc[-1][selected] > sma.iloc[-1][selected] 


        signaled_tickers = [l for l in target.temp['Signal'] if target.temp['Signal'][l] == True]
        #rdf = target.universe.loc[:target.now - self.lag - self.lookback,signaled_tickers].pct_change(1).dropna()
        rdf = target.universe.loc[: target.now,signaled_tickers].pct_change(1).dropna()

        for selected in target.perm['tickers']:
            if target.universe[selected].index[0] > t0:
                return False
            target.temp['Signal'][selected] = 0

        if len(rdf.index) > (self.lookback_months * 21) and len(signaled_tickers) > 1:
            optimal_weights = solve_weights(rdf)
            for count, selected in enumerate(signaled_tickers):
                target.temp['Signal'][selected] = optimal_weights[count]

        return True



class WeighFromSignal(bt.Algo):


    def __init__(self):
        super(WeighFromSignal, self).__init__()

    def __call__(self, target):

        target.temp['weights'] = {}
        for selected in target.perm['tickers']:
            if target.temp['Signal'][selected] is None:
                raise(Exception('No Signal!'))

            target.temp['weights'][selected] = target.temp['Signal'][selected]

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
            target.rebalance(item[1], child=item[0], base=base, update=False)

        # Now update
        target.root.update(target.now)

        return True
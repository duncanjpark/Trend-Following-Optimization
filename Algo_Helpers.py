import re
from tabnanny import verbose
import pandas as pd
from IPython.display import display
import bt
import numpy as np
import cvxpy as cp



def solve_weights(rdf, signals):
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

    try:
        result = prob.solve()
    except UserWarning:
        prob.solve(verbose=True)
    minvol = vol.value

    prob = cp.Problem(cp.Maximize(ret),  # maximize return
            [cp.sum(w) == 1, 
            w >= 0]
            )

    try:
        result = prob.solve()
    except UserWarning:
        prob.solve(verbose=True)
    maxretvol = vol.value

    max_vol = (maxretvol + minvol) / 2 
    prob = cp.Problem(cp.Maximize(ret),
                  [cp.norm1(w) <= 1.5,  # 1-norm <= 1.5, i.e. gross exposure < 150%
                   cp.sum(w) == 1,
                   vol <= max_vol]
                 )

    try:
        result = prob.solve()
    except UserWarning:
        prob.solve(verbose=True)
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
        if target.universe.index[0] > (target.now - self.lookback):
            return False
    
        prc = target.universe.loc[ target.now - self.lookback : , : ]

        sma  = prc.rolling(window=21*self.lookback_months,center=False).median().shift(self.lag_days)
        
        for selected in target.perm['tickers']:
            target.temp['Signal'][selected] = prc.iloc[-1][selected] / sma.iloc[-1][selected] - 1
            if np.isnan(target.temp['Signal'][selected]):
                target.temp['Signal'][selected] = 0
        best_signals = {k: v for k, v in sorted(target.temp['Signal'].items(), key = lambda item: item[1], reverse=True) }

        #signaled_tickers = [l for l in target.temp['Signal'] if abs(target.temp['Signal'][l]) > 0.003 ]
        signaled_tickers = [ l for l in best_signals.keys()][:int(len(best_signals.keys()) / 2)]
        rdf = target.universe.loc[target.now - self.lookback : target.now , signaled_tickers].pct_change(1).dropna()

        if  len(signaled_tickers) > 1:
            optimal_weights = solve_weights(rdf, target.temp['Signal'])
    

        if len(signaled_tickers) > 1:
            for count, ticker in enumerate(signaled_tickers):
                if (optimal_weights[count] * target.temp['Signal'][ticker]) > 0: #if same direction
                    target.temp['Signal'][ticker] = optimal_weights[count]
            for ticker in target.perm['tickers']:
                if ticker not in signaled_tickers:
                    target.temp['Signal'][ticker] = 0

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
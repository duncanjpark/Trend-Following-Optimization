import re
from tabnanny import verbose
import pandas as pd
from IPython.display import display
import bt
import numpy as np
import cvxpy as cp



def solve_weights(rdf, signals):
    # compute covariance matrix (df being the dataframe of historical returns)
    Sigma = np.cov(rdf.transpose())
    # number of assets
    n = Sigma.shape[0]
    # average returns
    mu = rdf.mean().values
    # asset SDs
    asset_vols = np.sqrt(Sigma.diagonal())
    # variable to optimize over - portfolio weights
    w = cp.Variable(n)
    # objectives to optimize
    # portfolio return
    ret = mu.T @ w 
    # volatility
    vol = cp.quad_form(w, Sigma)
    
    prob = cp.Problem(cp.Minimize(vol), # minimize volatility
        [#cp.norm1(w) <= 1.5,            # 1-norm <= 1.5, i.e. gross exposure < 150%
            cp.sum(w) == 1,              # sum of weights = 1
            w <= 0.25,
            w >= -.25
        ]
        )

    prob.solve()
    minvol = vol.value

    prob = cp.Problem(cp.Maximize(ret),  # maximize return
            [#cp.norm1(w) <= 1.5,         # 1-norm <= 1.5, i.e. gross exposure < 150%
            cp.sum(w) == 1,               # sum of weights = 1
            w >= -.25,
            w <= 0.25
            ]
            )

    prob.solve()
    maxretvol = vol.value

    max_vol = maxretvol * .5 + minvol * .5  #middle point between minimum vol and maximum for returns

    prob = cp.Problem(cp.Maximize(ret), #maximize returns
                  [cp.norm1(w) <= 1.5,  # 1-norm <= 1.5, i.e. gross exposure < 150%
                   cp.sum(w) == 1,      # sum of weights = 1
                   vol <= max_vol,      # use vol limit
                   w <= 0.25
                   ]
                 )


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
        # check that enough time has passed to look back
        if target.universe.index[0] > (target.now - self.lookback):
            return False
    
        #prices
        prc = target.universe.loc[ target.now - self.lookback : , : ]
        #SMA of prices with lag
        sma  = prc.rolling(window=21*self.lookback_months,center=False).median().shift(self.lag_days)
        
        #Set Signal to % difference of current vs lagged SMA
        for selected in target.perm['tickers']:
            target.temp['Signal'][selected] = prc.iloc[-1][selected] / sma.iloc[-1][selected] - 1
            if np.isnan(target.temp['Signal'][selected]):
                target.temp['Signal'][selected] = 0

        #sort the signals by this % change
        best_signals = {k: v for k, v in sorted(target.temp['Signal'].items(), key = lambda item: abs(item[1]), reverse=True) }
        #the tickers of these signals
        signaled_tickers = [ l for l in best_signals.keys()][:int(len(best_signals.keys()) * .25)]
        #daily returns of these signals in look back period
        rdf = target.universe.loc[target.now - self.lookback : target.now , signaled_tickers].pct_change(1).dropna()
        #rdf = target.universe.loc[ : target.now , signaled_tickers].pct_change(1).dropna()

        if  len(signaled_tickers) > 1:
            #Get the weightings of these tickers based off convex variance optimization 
            optimal_weights = solve_weights(rdf, target.temp['Signal'])

            for count, ticker in enumerate(signaled_tickers):
                #if (optimal_weights[count] * target.temp['Signal'][ticker]) > 0: #if both optimization and trend agree long or short
                target.temp['Signal'][ticker] = optimal_weights[count]       #then set weight of portfolio to that weight
            for ticker in target.perm['tickers']:
                if ticker not in signaled_tickers:
                    target.temp['Signal'][ticker] = 0                            #otherwise, liquidate holding / don't trade

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
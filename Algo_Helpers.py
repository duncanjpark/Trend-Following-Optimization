import pandas as pd
from IPython.display import display
import bt
import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt


def solve_weights(rdf, max_vol):
    labels = list(rdf.columns)
    Sigma = np.cov(rdf.transpose())
    n = Sigma.shape[0]
    mu = rdf.mean().values
    asset_vols = np.sqrt(Sigma.diagonal())
    w = cp.Variable(n)
    ret = mu.T @ w
    vol = cp.quad_form(w, Sigma)
    z = pd.DataFrame([mu, asset_vols], columns=labels)
    z['rows'] = ['real return', 'vol']
    z.set_index('rows')
    prob = cp.Problem(cp.Maximize(ret),
                  [cp.sum(w) == 1, 
                   w >= 0,
                   vol <= max_vol   # new constraint: vol <= vol_limit parameter
                  ]
                 )


    #vol_limit.value = vl_val
    result = prob.solve()
    return w.value
"""
    prob = cp.Problem(cp.Minimize(vol),
        [cp.sum(w) == 1,
        w >= 0]
        )

    prob.solve()
    wts = [float('%0.4f' % v) for v in w.value]
    minvol = vol.value

    pd.DataFrame([wts], columns=labels)

    prob = cp.Problem(cp.Maximize(ret),  # maximize return
            [cp.sum(w) == 1, 
            w >= 0]
            )

    prob.solve()
    wts = [float('%0.4f' % v) for v in w.value]
    maxretvol = vol.value

    vol_limit = cp.Parameter(nonneg=True)

    # define function so we can solve many in parallel
    def solve_vl(vl_val):
        vol_limit.value = vl_val
        result = prob.solve()
        return (ret.value, np.sqrt(vol.value), w.value)

    # number of points on the frontier
    NPOINTS = 200
    vl_vals = np.linspace(minvol, maxretvol, NPOINTS)

    # iterate in-process
    results_dict = {}
    for vl_val in vl_vals:
        # print(datetime.strftime(datetime.now(), "%H:%M:%S"), vl_val)
        results_dict[vl_val] = solve_vl(vl_val)

    ret_df = pd.DataFrame(enumerate(results_dict.keys()))
    ret_df.columns=['i', 'vol']
    ret_df['return'] = [results_dict[v][0] for v in ret_df['vol']]
    ret_df['std'] = [results_dict[v][1] for v in ret_df['vol']]
    for i, colname in enumerate(labels):
        ret_df[colname]=[results_dict[v][2][i] for v in ret_df['vol']]

    x = ret_df['return']
    # absolute values so shorts don't create chaos
    y_list = [abs(ret_df[l]) for l in labels]
"""
    #return ret_df

class Signal(bt.Algo):


    def __init__(self, lookback=pd.DateOffset(months=3),
                 lag=pd.DateOffset(days=0)):
        super(Signal, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):


        target.temp['Signal'] = {}
    
        for selected in target.perm['tickers']:

            t0 = target.now - self.lag

            if target.universe[selected].index[0] > t0:
                return False
            prc = target.universe[selected].loc[t0 - self.lookback:t0]

            trend = prc.iloc[-1]/prc.iloc[0] - 1
            signal = trend > 0.


            if signal:
                target.temp['Signal'][selected] = .2
            else:
                target.temp['Signal'][selected] = 0

        signaled_tickers = [l for l in target.temp['Signal'] if target.temp['Signal'][l] != 0]
        rdf = target.universe.loc[:target.now - self.lag - self.lookback,signaled_tickers]
        rdf = rdf.pct_change(1) # 1 for ONE DAY lookback
        rdf = rdf.dropna()

        for selected in target.perm['tickers']:
            target.temp['Signal'][selected] = 0

        if len(rdf.index) > 10 and len(signaled_tickers) > 1:
            max_vol = 0.0007
            optimal_info = solve_weights(rdf, max_vol)
            for count, selected in enumerate(signaled_tickers):
                target.temp['Signal'][selected] = optimal_info[count]


        

        
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
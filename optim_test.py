
import numpy as np
import pandas as pd
import bt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use( 'tkagg' )
#%matplotlib inline

from Algo_Helpers import Signal, WeighFromSignal, Rebalance
start_date = '2010-11-01'



pdf = bt.get('aapl,msft,c,ge,gs,gold,spy', start=start_date)
pdf = pdf.pct_change(1) # 1 for ONE DAY lookback
pdf = pdf.dropna()


import cvxpy as cp



labels = list(pdf.columns)


# compute covariance matrix
Sigma = np.cov(pdf.transpose())
# number of assets

n = Sigma.shape[0]
# average returns
mu = pdf.mean().values
# asset STDs
asset_vols = np.sqrt(Sigma.diagonal())
# variable to optimize over - portfolio weights
w = cp.Variable(n)

ret = mu.T @ w
vol = cp.quad_form(w, Sigma)


z = pd.DataFrame([mu, asset_vols], columns=labels)
z['rows'] = ['real return', 'vol']
z.set_index('rows')



prob = cp.Problem(cp.Minimize(vol),
                [cp.sum(w) == 1,
                w >= 0]
                )

prob.solve()
wts = [float('%0.4f' % v) for v in w.value]
minvol = vol.value

print("Min return portfolio weights")
pd.DataFrame([wts], columns=labels)


prob = cp.Problem(cp.Maximize(ret),  # maximize return
                  [cp.sum(w) == 1, 
                   w >= 0]
                 )
prob.solve()
wts = [float('%0.4f' % v) for v in w.value]
maxretvol = vol.value

print("Max return portfolio weights")
pd.DataFrame([wts], columns=labels)

# solve points in between
# maximize return subject to volatility constraints between minimum volatility and max return volatility

# specify a Parameter variable instead of creating new Problem at each iteration
# this allows the solver to reuse previous work
vol_limit = cp.Parameter(nonneg=True)

prob = cp.Problem(cp.Maximize(ret),
                  [cp.sum(w) == 1, 
                   w >= 0,
                   vol <= vol_limit
                  ]
                 )

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
ret_df


# plot efficient frontier
def plot_efrontier(ret_df, df,
                   xlabel="Standard Deviation of Real Returns",
                   ylabel="Real Return",
                   title=None):

    Sigma = np.cov(df.transpose())
    n = Sigma.shape[0]
    mu = df.mean().values
    asset_vols = np.sqrt(Sigma.diagonal())

    plt.figure(figsize=(8, 4.5))

    # plot the data
    plt.plot(ret_df['std'], ret_df['return'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plot_title = "Risk vs. Real Return"
    plt.title(plot_title)

    # plot the markers
    plt.scatter(asset_vols, mu)
    xoffset = 0.00015
    yoffset = 0.00002
    labels = df.columns
    #display(labels)
    for i, label in enumerate(labels):
        
        plt.annotate(label, xy=(asset_vols[i]+xoffset, mu[i]+yoffset),  xycoords='data',
                     horizontalalignment='left', verticalalignment='top',
                    )
        
plot_efrontier(ret_df, pdf)

plt.show()
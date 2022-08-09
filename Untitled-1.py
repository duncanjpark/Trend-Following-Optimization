#!/usr/bin/env python3

import matplotlib
import numpy as np
import pandas as pd
import bt
from IPython.display import display
import matplotlib.pyplot as plt
from Algo_Helpers import Signal, WeighFromSignal, Rebalance
matplotlib.use( 'tkagg' )


lookback_months = 2
lag_days = 1

pdf = pd.read_pickle(r'./pdf.pkl')


runMonthlyAlgo = bt.algos.RunWeekly()
rebalAlgo = Rebalance()

signalAlgo = Signal(lookback_months, lag_days)

weighFromSignalAlgo = WeighFromSignal()

s = bt.Strategy(
    'Trend Following with Mean Variance Optimization',
    [
        runMonthlyAlgo,
        signalAlgo,
        weighFromSignalAlgo,
        rebalAlgo
    ]
)
s.perm['tickers'] = list(pdf.columns)
t = bt.Backtest(s, pdf, integer_positions=False, progress_bar=True)
res = bt.run(t)

plt.plot(res.prices)

res.plot_security_weights()



df = pdf['aapl']

tmt = res.get_security_weights()['aapl']
tmt = tmt.rename("weight")
tmt = tmt.drop(tmt.index[0])
rdf = pd.concat([df, tmt], axis=1)
rdf['label'] = np.where(rdf['weight'] == 0, 0, np.where(rdf['weight'] < 0, -1, 1))
fig, ax = plt.subplots()
fig.set_size_inches(10,8)
def plot_func(group):
    global ax
    #color = 'r' if (group['label'] < 0).all() else 'g' if (group['label'] > 0).all() else 'y'
    lw = 1
    if (group.label ==-1).all() :
        color = 'r'
    elif (group.label == 1).all() :
        color ='g'
    elif (group.label == 0).all() :
        color='y'
    
    ax.plot(group.index, group.aapl, c=color, linewidth=lw)



rdf.groupby((rdf['label'].shift() != rdf['label']).cumsum()).apply(plot_func)

plt.show()



display(res.stats)




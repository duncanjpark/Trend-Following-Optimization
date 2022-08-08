#!/usr/bin/env python3

import matplotlib
import numpy as np
import pandas as pd
import bt
from IPython.display import display
import matplotlib.pyplot as plt
from Algo_Helpers import Signal, WeighFromSignal, Rebalance
matplotlib.use( 'tkagg' )

start_date = '2018-11-01'


pdf = bt.get('aapl,msft,c,ge,gs, nvda, dis, jnj, ibm, mrk, pg, rtx, csco', start=start_date)



runMonthlyAlgo = bt.algos.RunWeekly()
rebalAlgo = Rebalance()

signalAlgo = Signal(pd.DateOffset(months=2),pd.DateOffset(days=1))

weighFromSignalAlgo = WeighFromSignal()

s = bt.Strategy(
    'example1',
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
#pr = res.prices


res.plot_security_weights()
#res.prices.tail()
#temp = res.get_security_weights()
#test = res.get_transactions()
#q = t.positions


df = pdf['aapl']

tmt = res.get_security_weights()['aapl']
tmt = tmt.rename("weight")
tmt = tmt.drop(tmt.index[0])
rdf = pd.concat([df, tmt], axis=1)
rdf['label'] = np.where(rdf['weight'] == 0, -1, 1)


fig, ax = plt.subplots()
fig.set_size_inches(10,8)
def plot_func(group):
    global ax
    color = 'r' if (group['label'] < 0).all() else 'g'
    lw = 1
    ax.plot(group.index, group.aapl, c=color, linewidth=lw)


rdf.groupby((rdf['label'].shift() * rdf['label'] < 0).cumsum()).apply(plot_func)

plt.show()




res.stats




import pandas as pd
import ffn
import bt

def run_algo(data):
    sma = data.rolling(window=21*12, center=False).median().shift(1)

    trend = sma.copy()
    trend[data > sma] = True
    trend[data <= sma] = False
    trend[sma.isnull()] = False

    tsmom_invvol_strat = bt.Strategy(
        'tsmom_invvol',
        [
            bt.algos.RunDaily(),
            bt.algos.SelectWhere(trend),
            bt.algos.WeighInvVol(),
            bt.algos.LimitWeights(limit=0.4),
            bt.algos.Rebalance()
        ]
    )

    tsmom_invvol_bt = bt.Backtest(
        tsmom_invvol_strat,
        data,
        initial_capital=50000000.0,
        commissions=lambda q, p: max(100, abs(q) * 0.0021),
        integer_positions=False,
        progress_bar=True
    )
    tsmom_invvol_res = bt.run(tsmom_invvol_bt)

    return tsmom_invvol_res
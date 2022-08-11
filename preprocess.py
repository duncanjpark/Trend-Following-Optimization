import requests
import re
import bt
import pandas as pd

num_initial_tickers = 100
etf_key = 'SPY'
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
                  "Chrome/51.0.2704.103 Safari/537.36"
}

url = ("https://www.zacks.com/funds/etf/" + etf_key + "/holding")
with requests.Session() as req:
    req.headers.update(headers)
    r = req.get(url)
    etf_stock_list = re.findall(r'etf\\\/(.*?)\\', r.text)
    etf_stock_list = [x.replace('.', '-') for x in etf_stock_list]
    etf_stock_details_list = re.findall(
        r'<\\\/span><\\\/span><\\\/a>",(.*?), "<a class=\\\"report_[a-z]+ newwin\\', r.text)

    new_details = [x.replace('\"', '').replace(',', '').split() for x in etf_stock_details_list ]
    holdings = pd.DataFrame(new_details[:num_initial_tickers], index=etf_stock_list[:num_initial_tickers], columns=['Shares', 'Weight', '52 Wk Change(%)'])
    

tickers = list(holdings.index[:num_initial_tickers])

start_date = '2018-11-01'
pdf = bt.get(tickers, start=start_date)
pdf = pdf.dropna(axis='columns')


pdf.to_pickle(r'./pdf.pkl')
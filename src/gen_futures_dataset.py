import os
import pandas as pd
import argparse


INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'inputs')
FUTURES_DATA_PATH = os.path.join(INPUTS_PATH, 'pinnacle', 'CLCDATA')

parser = argparse.ArgumentParser(description='Generate futures dataset')
parser.add_argument('--continuous_future_method', type=str, default='RAD')
parser.add_argument('--output_path', type=str, default=INPUTS_PATH)

def generate_futures_dataset(tickers, continuous_future_method):

    tickers_data = []
    for ticker, flds in tickers.items():
        ticker_data = pd.read_csv(
            os.path.join(FUTURES_DATA_PATH, f'{ticker}_{continuous_future_method}.CSV'),
        )

        # name columns appropriately
        ticker_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest']

        # fix date format
        ticker_data['date'] = pd.to_datetime(ticker_data['date'], format='%m/%d/%Y')

        # select flds
        ticker_data = ticker_data[['date'] + flds]

        # set date as index
        ticker_data.set_index('date', inplace=True)

        # rename flds
        if len(flds) == 1:
            ticker_data.rename(columns={flds[0]: ticker}, inplace=True)
        else:
            ticker_data.rename(columns={fld: f'{ticker}_{fld}' for fld in flds}, inplace=True)

        tickers_data.append(ticker_data)
    tickers_data = pd.concat(tickers_data, axis=1)

    # resample data to business days and forward fill missing values
    tickers_data = tickers_data.resample('B').last().ffill()

    # get non-na intersection of all tickers
    tickers_data_nonan = tickers_data.copy().dropna(how='any')

    # ticker returns
    tickers_returns_daily = tickers_data_nonan.copy().pct_change().dropna()

    # ticker returns monthly
    tickers_data_nonan_monthly = tickers_data_nonan.resample('ME').last()
    tickers_returns_monthly = tickers_data_nonan_monthly.copy().pct_change().dropna()

    return tickers_returns_daily, tickers_returns_monthly

if __name__ == '__main__':
    args = parser.parse_args()
    continuous_future_method = args.continuous_future_method
    output_path = args.output_path

    # define tickers and fields
    tickers = {

        # commodities
        ## grains
        'ZC': ['close'], # corn
        'ZO': ['close'], # oats
        'ZL': ['close'], # soybean oil
        'ZR': ['close'], # rough rice
        'ZW': ['close'], # wheat
        ## softs
        'CC': ['close'], # cocoa
        'DA': ['close'], # milk
        'JO': ['close'], # orange juice
        'KC': ['close'], # coffee
        'SB': ['close'], # sugar
        ## livestock
        'LB': ['close'], # lumber
        'ZF': ['close'], # feeder cattle
        'ZZ': ['close'], # lean hogs
        'ZT': ['close'], # live cattle
        ## metals
        'ZG': ['close'], # gold
        'ZI': ['close'], # silver
        'ZK': ['close'], # copper
        'ZP': ['close'], # platinum
        ## energy
        'ZN': ['close'], # natural gas
        'ZU': ['close'], # crude oil
        
        # bonds
        'CB': ['close'], # 10-year canadian bond
        'DT': ['close'], # euro bund
        'EC': ['close'], # eurodollar
        'FB': ['close'], # 5-year us treasury bond
        'GS': ['close'], # gilt
        'TU': ['close'], # 2-year us treasury note
        'TY': ['close'], # 10-year us treasury note
        'UB': ['close'], # ultra us treasury bond
        # 'US': ['close'], # T-bonds
        'UZ': ['close'], # euro schatz
        
        # # fx
        'AN': ['close'], # audusd
        'CN': ['close'], # cadusd
        'BN': ['close'], # gbpusd
        'DX': ['close'], # us dollar index
        'JN': ['close'], # jpyusd
        'MP': ['close'], # mxnusd
        'SN': ['close'], # chfusd

        # # equities
        # 'FN': ['close'], # euro
        'NK': ['close'], # nikkei
        # 'CA': ['close'], # cac 40
        # 'EN': ['close'], # nasdaq mini
        # 'ER': ['close'], # russell 2000 mini
        'ES': ['close'], # e-mini s&p 500
        'LX': ['close'], # ftse 100
        # 'MD': ['close'], # S&P 400 midcap
        'XU': ['close'], # euro stoxx 50

        # 'XU': ['close'], # euro stoxx 50
        # 'ES': ['close'], # e-mini s&p 500
        # 'TY': ['close'], # 10-year us treasury note
        # 'FB': ['close'], # 5-year us treasury bond
        # 'ZG': ['close'], # gold
        # 'ZI': ['close'], # silver
        # 'ZN': ['close'], # natural gas
        # 'ZU': ['close'], # crude oil

    }

    tickers_returns_daily, tickers_returns_monthly = generate_futures_dataset(tickers, args.continuous_future_method)

    # save to csv
    tickers_returns_daily.to_csv(os.path.join(args.output_path, 'futures_dataset_daily.csv'))
    tickers_returns_monthly.to_csv(os.path.join(args.output_path, 'futures_dataset_monthly.csv'))

    print(f"Futures dataset generated with {len(tickers)} tickers using {continuous_future_method} method.")
    print(f"Daily returns saved to {os.path.join(args.output_path, 'futures_dataset_daily.csv')}")
    print(f"Monthly returns saved to {os.path.join(args.output_path, 'futures_dataset_monthly.csv')}")


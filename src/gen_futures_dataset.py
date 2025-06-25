import os
import pandas as pd
import argparse


INPUTS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'inputs')
FUTURES_DATA_PATH = os.path.join(INPUTS_PATH, 'pinnacle', 'CLCDATA')

parser = argparse.ArgumentParser(description='Generate futures dataset')
parser.add_argument('--continuous_future_method', type=str, default='RAD')
parser.add_argument('--output_path', type=str, default=os.path.join(INPUTS_PATH, 'futures_dataset.csv'))

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
    tickers_returns = tickers_data_nonan.pct_change().dropna()

    return tickers_returns

if __name__ == '__main__':
    args = parser.parse_args()
    continuous_future_method = args.continuous_future_method
    output_path = args.output_path

    # define tickers and fields
    tickers = {

        # commodities
        'ZG': ['close'], 'ZK': ['close'], 'ZU': ['close'],

        # bonds
        'ZB': ['close'], 'ZC': ['close'], 'ZF': ['close'], 'ZT': ['close'],

        # fx
        'FN': ['close'], 'BN': ['close'], 'CN': ['close'], 'AN': ['close'], 'JN': ['close'],

        # equities
        'ES': ['close'], 'XU': ['close'], 'NK': ['close'], 'LX': ['close'],

    }

    tickers_returns = generate_futures_dataset(tickers, args.continuous_future_method)

    # save to csv
    tickers_returns.to_csv(args.output_path)


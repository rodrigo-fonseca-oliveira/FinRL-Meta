from finrl.meta.data_processor import DataProcessor
import pandas as pd

API_KEY = ""
API_SECRET = ""
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'


import numpy as np
import pandas as pd
from tqdm import tqdm
import exchange_calendars as tc


def get_trading_days(start, end):
    nyse = tc.get_calendar("NYSE")
    df = nyse.sessions_in_range(
        pd.Timestamp(start), pd.Timestamp(end)
    )
    trading_days = []
    for day in df:
        trading_days.append(str(day)[:10])

    return trading_days

def clean_data(df):
    """
    Cleans the provided dataframe to ensure:
    - All tickers have data for all timestamps.
    - Missing data is filled accordingly.
    
    Args:
    - df (DataFrame): The input dataframe with ticker data.

    Returns:
    - DataFrame: The cleaned dataframe.
    """

    def fill_with_first_valid(tmp_df, column):
        """
        Fills the first NaN entry of a DataFrame with the first valid entry for a given column.

        Args:
        - tmp_df (DataFrame): The DataFrame to modify.
        - column (str): The column to consider.
        
        Returns:
        - DataFrame: The modified DataFrame.
        """
        first_valid = tmp_df[column].first_valid_index()
        if first_valid is not None and first_valid != tmp_df.index[0]:
            first_valid_value = tmp_df.at[first_valid, column]
            tmp_df.iloc[0] = [first_valid_value] * 4 + [0.0]
        return tmp_df

    tic_list = df['tic'].unique()
    n_tickers = len(tic_list)

    # Filter out timestamps that don't have data for all tickers
    df = df.groupby("timestamp").filter(lambda x: len(x) == n_tickers)

    trading_days = get_trading_days(start=df.timestamp.min(), end=df.timestamp.max())
    NY = "America/New_York"
    times = [pd.Timestamp(day + " 09:30:00").tz_localize(NY) + pd.Timedelta(minutes=15*i) for day in trading_days for i in range(390)]

    # Create a new dataframe with full timestamp series
    new_df = pd.DataFrame()
    for tic in tqdm(tic_list):
        tmp_df = pd.DataFrame(index=times).assign(open=np.nan, high=np.nan, low=np.nan, close=np.nan, volume=np.nan)
        tic_df = df[df.tic == tic].set_index('timestamp')
        tmp_df.update(tic_df)

        if pd.isna(tmp_df.iloc[0]['close']):
            print(f"The price of the first row for ticker {tic} is NaN.")
            tmp_df = fill_with_first_valid(tmp_df, 'close')
            if pd.isna(tmp_df.iloc[0]['close']):
                print(f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0.")
                tmp_df.iloc[0] = [0.0] * 5

        # Forward fill NaN values
        tmp_df.fillna(method='ffill', inplace=True)

        tmp_df = tmp_df.astype(float)
        tmp_df['tic'] = tic
        new_df = pd.concat([new_df, tmp_df])

    new_df = new_df.reset_index().rename(columns={"index": "timestamp"})

    return new_df

dp = DataProcessor(data_source = 'alpaca',
                  API_KEY = API_KEY, 
                  API_SECRET = API_SECRET, 
                  API_BASE_URL = API_BASE_URL
                  )

# data = dp.download_data(start_date = TRAIN_START_DATE, 
#                         end_date = TRAIN_END_DATE,
#                         ticker_list = ticker_list, 
#                         time_interval= '1Min')
# data.to_hdf("dow_30_intraday.h5", key="data")

import time

start_time = time.time()  #

print("start reading data stored")
with pd.HDFStore("dow_30_intraday.h5") as store:
    data = store.get("data")
print("cleaning data")
data = clean_data(data)
print("adding technical indicators")
data = dp.add_technical_indicator(data, technical_indicator_list)
print("adding vix"  )
data = dp.add_vix(data)
print("store model data")
data.to_hdf("dow_30_intraday.h5", key="model_data")

end_time = time.time()  # Capture end time

elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.5f} seconds")
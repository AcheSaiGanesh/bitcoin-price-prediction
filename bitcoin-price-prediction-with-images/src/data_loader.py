import pandas as pd

def load_csv(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df = df.reset_index(drop=True)
    return df

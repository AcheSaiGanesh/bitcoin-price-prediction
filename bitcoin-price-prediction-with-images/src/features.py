import numpy as np
import pandas as pd

def add_technical_indicators(df):
    df = df.copy()
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma21'] = df['close'].rolling(window=21).mean()
    df['std21'] = df['close'].rolling(window=21).std()
    df['returns'] = df['close'].pct_change()
    df = df.dropna()
    return df

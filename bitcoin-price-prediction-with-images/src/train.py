import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_csv
from src.features import add_technical_indicators
from src.models import build_lstm
from sklearn.preprocessing import StandardScaler

def prepare_sequences(df, seq_len=60):
    data = df[['close','ma7','ma21','std21','returns']].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i])
        y.append(data_scaled[i,0])  # predict scaled close
    return np.array(X), np.array(y), scaler

def run_training(csv_path):
    df = load_csv(csv_path)
    df = add_technical_indicators(df)
    X, y, scaler = prepare_sequences(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save('lstm_model.h5')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python src/train.py data/bitcoin_prices.csv')
    else:
        run_training(sys.argv[1])

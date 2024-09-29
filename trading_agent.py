import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AITradingAgent:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()

    def train(self, data):
        data['lag_1'] = data['close'].shift(1)
        data['lag_2'] = data['close'].shift(2)
        data = data.dropna()

        X = data[['lag_1', 'lag_2']]
        y = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # 1 = Buy, 0 = Sell
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)

    def predict(self, row):
        X_new = self.scaler.transform([[row['lag_1'], row['lag_2']]])
        prediction = self.model.predict(X_new)
        return "BUY" if prediction == 1 else "SELL"

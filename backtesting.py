import pandas as pd
import numpy as np

# Backtesting class for evaluating strategies
class Backtester:
    def __init__(self, market_data, model):
        self.data = market_data
        self.model = model
        self.balance = 10000  # Initial capital
        self.position = 0  # Current position (1 for long, -1 for short)
        self.trades = []

    def simulate(self):
        for index, row in self.data.iterrows():
            input_data = row[['open', 'high', 'low', 'volume']].values.reshape(1, -1)
            prediction = self.model.predict(input_data)

            # Example trading logic
            if prediction > row['close'] and self.position != 1:
                self.buy(row['close'])
            elif prediction < row['close'] and self.position != -1:
                self.sell(row['close'])

    def buy(self, price):
        if self.position == -1:
            self.balance += (price - self.entry_price) * self.position_size
        self.position = 1
        self.entry_price = price
        self.position_size = self.balance / price
        self.trades.append(('buy', price))

    def sell(self, price):
        if self.position == 1:
            self.balance += (price - self.entry_price) * self.position_size
        self.position = -1
        self.trades.append(('sell', price))

    def performance_metrics(self):
        # Example performance metrics
        total_trades = len(self.trades)
        profit = self.balance - 10000
        sharpe_ratio = np.mean([trade[1] for trade in self.trades]) / np.std([trade[1] for trade in self.trades])
        return {
            "Total Trades": total_trades,
            "Profit": profit,
            "Sharpe Ratio": sharpe_ratio
        }

# Example of using the backtester
if __name__ == "__main__":
    market_data = pd.read_csv('market_data.csv')  # Load historical data
    model = train_model(market_data)  # Train your model
    backtester = Backtester(market_data, model)
    backtester.simulate()
    metrics = backtester.performance_metrics()
    print(metrics)

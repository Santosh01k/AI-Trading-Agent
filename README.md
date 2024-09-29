# AI-Trading-Agent
## Overview
This project implements an AI trading agent that autonomously performs trades on the Orderly Network by leveraging machine learning algorithms to predict market trends and execute trades. The agent is built using the **Empyreal SDK** and interacts with the **Orderly Network SDK** to perform live trading. The project also includes a backtesting tool to evaluate the agent's performance.

## Features
- **Machine Learning Integration**: Utilizes a Random Forest Regressor to predict future price movements based on historical market data.
- **Order Execution**: Automatically places buy/sell orders on the Orderly Network using the SDK.
- **Backtesting Framework**: Simulate the AI agent’s performance on historical data with risk metrics like Sharpe Ratio and profit.
- **Comprehensive Risk Management**: Includes basic risk management features and can be extended to include advanced strategies like stop-loss and take-profit mechanisms.

## Tools & Technologies
- **Empyreal SDK**: For building the AI agent’s trading logic.
- **Orderly Network SDK**: To interact with the decentralized order book framework and execute trades.
- **Google Cloud Platform (GCP)**: Optional integration for advanced analytics using Vertex AI, BigQuery, and Compute Engine.
  
## Getting Started

### Prerequisites
- Python 3.8+
- Install the required dependencies by running:
  
```bash
pip install -r requirements.txt
```

## How to Run the AI Trading Agent

**1. Training the Model:** The AI agent trains on historical market data to predict future price movements.

**2. Executing Trades:** Based on predictions, the agent places buy/sell orders on the Orderly Network.

```bash
python trading_agent.py

```
## How to Run Backtesting
You can evaluate the strategy’s performance on historical market data using the backtesting module.

```bash
python backtesting.py
```

## Project Structure
The project is organized as follows:
```bash
.
├── README.md               # Project documentation
├── trading_agent.py        # Core trading logic of the AI agent
├── backtesting.py          # Script for backtesting the AI agent's performance
├── requirements.txt        # Python dependencies
├── .env                    # Contains sensitive information like API keys (not included in repo)
└── models/
    └── trained_model.pkl   # Trained machine learning model (optional)
```
## Detailed File Descriptions:
**trading_agent.py:** 
  Fetches live market data from the Orderly Network.Trains a Random Forest model on historical data.Executes buy/sell trades based on the predicted price movements.
  
**backtesting.py:**
Simulates the trading agent's performance on historical market data.Reports key metrics such as profit, Sharpe Ratio, and total trades.

**requirements.txt:**
Contains all Python libraries needed to run the project, including pandas, numpy, and scikit-learn.

**data/market_data.csv:**
Historical market data for backtesting. You can update this with your own data.

**models/trained_model.pkl:**
Optional file for storing the pre-trained machine learning model (if you want to load a pre-trained model for faster execution).

## Future Work

**Advanced Machine Learning:** Experimenting with more complex models like neural networks or deep learning for more accurate predictions.

**Real-time Data Processing:** Use GCP tools like Pub/Sub and DataFlow for real-time data ingestion and processing.

**Risk Management:** Implementing advanced risk strategies such as stop-loss orders, take-profit, and portfolio optimization.

## Conclusion
The AI trading agent represents a significant step forward in automating trading strategies using machine learning and decentralized infrastructure. By leveraging both the Empyreal SDK and Orderly Network SDK, the agent can efficiently predict market trends and autonomously execute trades



# 💹 Trading Agent

The AI Trading Agent Dashboard is an intelligent web-based simulation platform designed to help users explore algorithmic trading strategies powered by Reinforcement Learning (DQN) and LSTM-based price prediction. It combines cutting-edge machine learning models with an intuitive interface to provide insights, visualizations, and performance metrics — all in real time. 📈🤖

🚀 Key Features

📊 Real-time Trading Simulation

Simulate stock trading with a Deep Q-Network agent trained on historical market data. Users can set an initial balance and evaluate the model's performance.

🧠 LSTM-Based Price Prediction

Predicts future prices using the last 60 days of market data, helping the agent make smarter decisions.

🎯 Reinforcement Learning Agent (DQN)

Implements a Dueling Deep Q-Network (DQN) trained to maximize long-term profit through intelligent buy/hold/sell strategies.

📈 Interactive Charts

Visualize portfolio value, actual vs. predicted stock prices, and trading actions over time using Chart.js.

📅 Date-wise Simulation

Track performance over specific date ranges with visual markers for each agent action.

⚙️ Customizable Parameters

Set your own initial balance to test different trading strategies and financial conditions.

📦 Modular Architecture

environment/TradingENV.py - Trading environment logic

environment/DQNAgent.py & DuelingDQN.py - Reinforcement learning agent

environment/LSTM_model.py - LSTM-based price predictor

app.py - Flask web interface

🖼️ Interface Preview

🎛️ Dashboard Simulation Panel

![ab084f85-995c-4cc4-8ed0-b712961da1a4](https://github.com/user-attachments/assets/38559af9-e3cd-4bc0-9a43-4155a196fab6)

📉 Trading Decisions Chart

![f79c7d3e-7be4-4e67-a240-107ead058f28](https://github.com/user-attachments/assets/7935ee0e-6477-4ce7-8241-a2c7a3700067)

💰 Portfolio Value Over Time

![3e380932-4f47-4d54-8637-5979b2824cab](https://github.com/user-attachments/assets/af6c09fa-1030-4bc9-bbbe-35acdf1d32b3)





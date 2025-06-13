import pandas as pd
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from environment.TradingENV import TradingENV
from environment.DQNAgent import DQNAgent
from environment.DuelingDQN import DuelingDQN
from environment.LSTM_model import LSTMModel
from sklearn.preprocessing import MinMaxScaler

lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load("prediction_model.pth"))
lstm_model.eval()
scaler = MinMaxScaler(feature_range=(0,1))

app = Flask(__name__)

# Load the trained model
def load_model(model_path, state_dim, action_dim):
    model = DuelingDQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Get LSTM prediction for 60 days of data
def get_lstm_prediction(last_60_days, scaler):
    last_60_days = scaler.fit_transform(last_60_days)
    last_60_days = torch.tensor(last_60_days, dtype=torch.float32)
    last_60_days = last_60_days.view(1, 60, 1)
    with torch.no_grad():
        predicted_price = lstm_model(last_60_days).item()
    return scaler.inverse_transform(np.array(predicted_price).reshape(-1, 1))[0][0]

# Run a test episode
def run_test_episode(env, agent):
    state = env.reset()
    done = False
    
    # Store predictions separately
    predictions = []
    
    # Get initial 60 days for first prediction
    if hasattr(env, 'get_60_days'):
        last_60_days = env.get_60_days()
        predictions.append(get_lstm_prediction(last_60_days, scaler))

    while not done:
        # Get action from agent
        with torch.no_grad():
                state = torch.FloatTensor(state).to(agent.device)  # Prepare the state tensor
                q_values = agent.model(state.unsqueeze(0)) # Get Q-values from the network
                action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        
        # Store prediction if available
        if hasattr(env, 'prediction_cache') and env.prediction_cache is not None:
            if isinstance(env.prediction_cache['prediction'], np.ndarray):
                predictions.append(env.prediction_cache['prediction'][0][0])
            else:
                predictions.append(env.prediction_cache['prediction'])
        
        state = next_state

    # Make sure predictions match prices in length
    while len(predictions) < len(env.prices) - 1:  # -1 to account for initial price
        predictions.append(predictions[-1] if predictions else env.prices[0][0])

    results = {
        'prices': env.prices,
        'actions': env.actions_taken[1:],  # Skip the first action
        'portfolio_values': env.portfolio_values,
        'final_balance': env.balance,
        'shares_hold': env.shares_hold,
        'total_profit': env.total_profit,
        'max_drawdown': env.max_drawdown,
        'predictions': predictions
    }
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.get_json()
    initial_balance = data.get('initial_balance', 10000)

    df = pd.read_csv("normalized_data_withFeatureAndDate.csv")

    env = TradingENV(df, initial_balance)
    state_dim = env.observation_space.shape[0]
    action_dim = 3

    agent = DQNAgent(state_dim, action_dim)
    model = load_model('trial_with.pth', state_dim, action_dim).to(agent.device)
    agent.model = model
    agent.epsilon = 0  # No exploration during testing

    results = run_test_episode(env, agent)

    # Prepare data for Chart.js
    prices = [item[0] for item in results['prices'][1:]]  # Skip initial price
    dates = [item[1] for item in results['prices'][1:]]
    
    chart_data = {
        'labels': dates,
        'prices': prices,
        'predictions': results['predictions'],
        'actions': results['actions'],
        'portfolio_values': results['portfolio_values']
    }

    return jsonify({
        'chart_data': chart_data,
        'final_balance': f"${results['final_balance']:.2f}",
        'shares_hold': results['shares_hold'],
        'total_profit': f"${results['total_profit']:.2f}",
        'profit_percentage': f"{(results['total_profit'] / initial_balance) * 100:.2f}%",
        'max_drawdown': f"{results['max_drawdown'] * 100:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
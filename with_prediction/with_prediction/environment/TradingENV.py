import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from environment.LSTM_model import LSTMModel
from sklearn.preprocessing import MinMaxScaler
CLOSE = 'Close'
DATE = 'Date'
class TradingENV(gym.Env):
    def __init__(self,data,initial_balance = 10000):
        super(TradingENV,self).__init__()

        #Trading dataset
        self.data = data
        self.current_steps = 0

        #initial balance
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.shares_hold = 0
        self.entry_price = 0
        self.trading_fees = 0.001
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10

        self.total_profit = 0
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0
         
        # max steps before truncating the episode
        self.max_steps = 25
        self.normalized_cols = ['feature_prevclose','feature_open','feature_high','feature_low','feature_last','feature_close','feature_vwap']
        
        # choosing random starting point
        self.start_random = np.random.randint(61,len(self.data) - 61)
        #observation space normalized data ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP'] 
        self.observation_space = spaces.Box(low=-1, high=1, shape=(len(self.normalized_cols) + 5,), dtype=np.float32)

        #action space (0 = hold, 1 = buy, 2 = sell)
        self.action_space = spaces.Discrete(3)
        #saving actions taken for plotting the graph
        self.actions_taken = [0]
        self.prices = [[self.data.iloc[self.get_index()][CLOSE],self.data.iloc[self.get_index()][DATE]]]
        self.portfolio_values = []
        self.prediction_list = []

        self.path = ""
        self.lstm_model = LSTMModel()
        self.lstm_model.load_state_dict(torch.load("prediction_model.pth"))
        self.lstm_model.eval()

        self.prediction_cache = None
        self.scaler = MinMaxScaler(feature_range=(0,1))


    def get_index(self):
        return (self.start_random + self.current_steps) % len(self.data)
    

    def reset(self):
        self.start_random = np.random.randint(61,len(self.data) - 61)
        self.current_steps = 0
        self.balance = self.initial_balance
        self.shares_hold = 0
        self.total_profit = 0
        self.entry_price = 0
        self.actions_taken = [0]
        self.prices = [[self.data.iloc[self.get_index()][CLOSE],self.data.iloc[self.get_index()][DATE]]]
        self.portfolio_values = []
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        self.prediction_list =[]
        

        return self.get_state()
    

    def get_reward(self, current_portfolio_value,action):

        self.portfolio_values.append(current_portfolio_value)
        
        
        if current_portfolio_value < self.max_portfolio_value:
            drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value
            self.max_drawdown = max(self.max_drawdown,drawdown)
        
        sell_bonus = 0
        if action == 2 and self.shares_hold > 0:  # Sell
            profit_percentage = (self.data.iloc[self.get_index()][CLOSE] - self.entry_price) / self.entry_price
            if profit_percentage > 0:
                sell_bonus = profit_percentage * 16  # Encourage profitable sells

        if action == 2  and self.shares_hold == 0:
            sell_bonus = -0.2

        # Penalize buying if already holding stocks
        buy_penalty = -0.5 if action == 1 and self.shares_hold > 0 else 0

        # Discourage doing nothing for too long
        hold_penalty = -0.02 if action == 0 else 0


        prediction = self.prediction_cache['prediction']
        current_price = self.data.iloc[self.get_index()][CLOSE]
        prediction_consideration = 0
        if (action == 1 and prediction > current_price) or (action == 2 and prediction < current_price):
            prediction_consideration = 0.1 

        
        self.max_portfolio_value = max(self.max_portfolio_value, current_portfolio_value)

        reward = (current_portfolio_value - self.initial_balance) / self.initial_balance
        risk_penalty = self.max_drawdown * 2

        return reward - risk_penalty + sell_bonus + buy_penalty + hold_penalty + prediction_consideration
    
    def get_lstm_prediction(self,last_60_days):
        
        last_60_days = self.scaler.fit_transform(last_60_days)

        last_60_days = torch.tensor(last_60_days, dtype=torch.float32)
        last_60_days = last_60_days.view(1,60,1)
        with torch.no_grad():
            
            predicted_price = self.lstm_model(last_60_days).item()

        return self.scaler.inverse_transform(np.array(predicted_price).reshape(-1,1))
    

    def get_60_days(self):
        last_60_days = []
        curr_index = self.get_index()
        if curr_index - 60 < 0:
            last_60_days = [0] *(60-curr_index)
            last_60_days.extend((self.data.iloc[curr_index:60][CLOSE].values))

        else:
            last_60_days.extend(self.data.iloc[curr_index:curr_index+60][CLOSE].values)

        return np.array(self.data.iloc[curr_index:curr_index+60][CLOSE].values).reshape(-1,1)
    def step(self,action):

        self.current_steps += 1

        if self.current_steps >= self.max_steps:
            done = True
            self.render()

            return np.zeros(len(self.normalized_cols) + 5), self.total_profit, done, {}
        

        new_price = self.data.iloc[self.get_index()][CLOSE]
        date = self.data.iloc[self.get_index()][DATE]
        self.prices.append([new_price,date])
        

        predicted_price = self.get_lstm_prediction(self.get_60_days())
        self.prediction_list.append(predicted_price)
        self.prediction_cache = {
            'prediction' : predicted_price,
        }


        


        if action == 1:
            if self.balance > 0:
                shares_bought = self.balance // new_price
                transaction_cost = shares_bought * new_price * self.trading_fees
                if shares_bought > 0:

                    self.shares_hold += shares_bought
                    self.balance -= (shares_bought * new_price + transaction_cost)
                    self.entry_price = new_price
                    self.actions_taken.append(action)
                else:
                    self.actions_taken.append(0)
            else:
                self.actions_taken.append(0)

            

        elif action == 2:
            if self.shares_hold > 0:
                transaction_cost = self.shares_hold * new_price * self.trading_fees
                self.balance += (self.shares_hold * new_price - transaction_cost)
                self.shares_hold = 0

                profit_percentage = (new_price - self.entry_price) / self.entry_price
                self.actions_taken.append(action)
            else:
                self.actions_taken.append(0)
        else:
            self.actions_taken.append(action)
        



        current_portfolio_value = self.balance + (self.shares_hold * new_price)


        self.total_profit = self.balance + (self.shares_hold * new_price) - 10000
        


        reward = self.get_reward(current_portfolio_value,action)   # Profit-based reward
        done = (
            self.current_steps >= self.max_steps or 
            current_portfolio_value <= self.initial_balance * 0.5  # 50% loss threshold
        )
        print(reward,end=" ")
        
            
        
        return self.get_state(), reward, done, {}
    

    def render(self):
        current_price = self.data.iloc[self.get_index()][CLOSE]
        portfolio_value = self.balance + (self.shares_hold * current_price)
        print(f"""
        Step: {self.current_steps}
        Balance: {self.balance:.2f}
        Shares Held: {self.shares_hold}
        Portfolio Value: {portfolio_value:.2f}
        Current Price: {current_price:.2f}
        Max Drawdown: {self.max_drawdown:.2%}
        """)

    def plot_buy_sell(self):
        closing_prices = self.prices[1:]
        plt.plot(self.prices[1:])
        buy_signals = [index for index,value in enumerate(self.actions_taken[1:]) if value == 1]  # Indices where buy happened
        sell_signals = [index for index,value in enumerate(self.actions_taken[1:]) if value == 2] 
        print(len(closing_prices),len(self.actions_taken))
        plt.scatter(buy_signals, [closing_prices[i] for i in buy_signals], color='green', label='Buy', marker='o')
        plt.scatter(sell_signals, [closing_prices[i] for i in sell_signals], color='blue', label='Sell', marker='o')
        plt.legend()
        plt.show()

    def get_state(self):

        current_index = self.get_index()
        normalized_features = self.data[self.normalized_cols].iloc[current_index].values
        next_predicted_price = self.prediction_cache['prediction'] if self.prediction_cache else 0
        current_price = self.data.iloc[current_index][CLOSE]  
        portfolio_value_ratio = (
            (self.balance + (self.shares_hold * self.data.iloc[current_index][CLOSE])) / self.initial_balance
        )
        shares_ratio = self.shares_hold / (self.initial_balance / self.data.iloc[current_index][CLOSE])
        if type(next_predicted_price) == type(np.array([1])):
            next_predicted_price = next_predicted_price[0][0]

        return np.concatenate([
            normalized_features, 
            [portfolio_value_ratio, shares_ratio, self.max_drawdown,next_predicted_price,current_price]
        ])
    
    def save_runs(self):
        run_dir = "../runs"
        time_t = time.time()
        path = run_dir + "/run_" + str(time_t)
        os.mkdir(path)
        return path

    def save_episode(self):
        epsiode_path = self.path + "/ep" + str(time.time())
        os.mkdir(epsiode_path)
        np.save(epsiode_path + "/prices.npy", self.prices)
        np.save(epsiode_path + "/actions.npy",self.actions_taken)


    def plot_actual_predicted_with_signals(self,prices, predictions, actions):
        """
        prices: List of [price, date] - actual prices
        predictions: List of predicted prices (same length as prices)
        actions: List of actions taken (0=hold, 1=buy, 2=sell)
        """
        
        # Extract actual prices and dates
        actual_prices = [p[0] for p in prices[1:]]  # Skipping initial dummy price
        dates = [p[1] for p in prices[1:]]
        print(actual_prices,dates,len(actual_prices),len(dates))
        dates = np.array(dates).squeeze()
        predictions = np.array(predictions).squeeze()


        # Ensure predictions length matches prices
        if len(predictions) != len(actual_prices):
            predictions = predictions[:len(actual_prices)]

        # Identify buy and sell points
        buy_signals = [i for i, action in enumerate(actions[1:]) if action == 1]
        sell_signals = [i for i, action in enumerate(actions[1:]) if action == 2]

        plt.figure(figsize=(16,8))
        plt.plot(actual_prices, label='Actual Close Price', color='black')
        plt.plot(predictions, label='Predicted Close Price', color='orange', linestyle='--')

        # Mark buy and sell points
        plt.scatter(buy_signals, [actual_prices[i] for i in buy_signals], marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(sell_signals, [actual_prices[i] for i in sell_signals], marker='v', color='red', label='Sell Signal', s=100)

        plt.title('Actual vs Predicted Prices with Buy/Sell Signals')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()







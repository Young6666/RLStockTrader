import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf

class StockTradingEnv(gym.Env):
    def __init__(self, ticker='AAPL', start_date='2010-01-01', end_date='2021-12-31', initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        # 1. 데이터 다운로드 및 전처리
        print(f"Loading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        self.df = df.reset_index()

        if len(self.df) < 60:
            raise ValueError("Data too short for technical indicators.")

        # --- 기술적 지표 추가 ---
        # 1. 이동평균선 (SMA)
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_60'] = self.df['Close'].rolling(window=60).mean()
        
        # 2. RSI (상대강도지수)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (추세 지표) 

        exp12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = exp12 - exp26
        
        # NaN 제거
        self.df = self.df.dropna().reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        
        # [핵심 수정 부분] shape=(6,)으로 설정 (MACD 포함)
        # [0:자산비율, 1:주식비율, 2:RSI, 3:SMA20, 4:SMA60, 5:MACD]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
        # 데이터 정규화
        cash_ratio = self.balance / self.max_net_worth
        stock_value = self.shares_held * row['Close']
        stock_ratio = stock_value / self.max_net_worth
        
        rsi = row['RSI'] / 100.0
        sma20_ratio = row['Close'] / row['SMA_20']
        sma60_ratio = row['Close'] / row['SMA_60']
        macd_val = row['MACD'] / 5.0 # MACD 정규화
        
        # 6개의 값을 반환
        obs = np.array([cash_ratio, stock_ratio, rsi, sma20_ratio, sma60_ratio, macd_val], dtype=np.float32)
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        
        # 거래 로직 (분할 매매)
        if action == 1: # Buy
            invest_amount = self.balance * 0.5
            if invest_amount > current_price:
                shares_bought = int(invest_amount / current_price)
                self.shares_held += shares_bought
                self.balance -= shares_bought * current_price
                
        elif action == 2: # Sell
            if self.shares_held > 0:
                shares_sold = int(self.shares_held * 0.5)
                if shares_sold == 0 and self.shares_held > 0:
                    shares_sold = self.shares_held
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 보상 함수 (샤프 지수 개념 도입)
        # 0으로 나누기 방지
        safe_prev_net_worth = prev_net_worth if prev_net_worth > 0 else 1.0
        portfolio_return = (self.net_worth - safe_prev_net_worth) / safe_prev_net_worth
        
        reward = portfolio_return * 100 
        
        # 하락장 방어 보상 (Cash is King)
        if current_price < row['SMA_60'] and (self.balance / self.net_worth) > 0.5:
             reward += 0.1
             
        return self._get_obs(), reward, terminated, truncated, {'net_worth': self.net_worth}
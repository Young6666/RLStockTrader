import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf

class StockTradingEnv(gym.Env):
    def __init__(self, ticker='AAPL', start_date='2018-01-01', end_date='2022-01-01', initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        # 1. 데이터 다운로드 및 전처리
        print(f"Loading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        self.df = df.reset_index()

        if len(self.df) < 60:
            raise ValueError("Data too short for technical indicators.")

        # --- [핵심] 기술적 지표(Feature Engineering) 추가 ---
        # 1. 이동평균선 (SMA) 20일, 60일
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_60'] = self.df['Close'].rolling(window=60).mean()
        
        # 2. RSI (상대강도지수) - 매수/매도 타이밍 핵심 지표
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. NaN 제거 (지표 계산 초기 구간)
        self.df = self.df.dropna().reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        
        # Observation Space 수정: 의미 있는 지표만 전달
        # [0: 보유현금비율, 1: 보유주식가치비율, 2: RSI, 3: 가격/SMA20(이격도), 4: 가격/SMA60]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

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
        
        # 데이터 정규화 (Normalization) - 아주 중요!
        # 모든 값을 0~1 또는 비율(Ratio)로 변환하여 AI가 크기에 압도되지 않게 함
        
        # 1. 자산 상태 비율
        cash_ratio = self.balance / self.max_net_worth
        stock_value = self.shares_held * row['Close']
        stock_ratio = stock_value / self.max_net_worth
        
        # 2. 기술적 지표 (이미 비율이거나 0~100 사이 값)
        rsi = row['RSI'] / 100.0  # 0~1 사이로 변환
        sma20_ratio = row['Close'] / row['SMA_20'] # 1.0보다 크면 상승세
        sma60_ratio = row['Close'] / row['SMA_60'] # 1.0보다 크면 장기 상승세
        
        obs = np.array([cash_ratio, stock_ratio, rsi, sma20_ratio, sma60_ratio], dtype=np.float32)
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        
        # 거래 로직 (보유 현금의 50% 매수 / 보유 주식의 50% 매도) - 분할 매매 도입
        if action == 1: # Buy
            # 올인하지 않고 현금의 절반만 사용 (리스크 관리)
            invest_amount = self.balance * 0.5
            if invest_amount > current_price:
                shares_bought = int(invest_amount / current_price)
                self.shares_held += shares_bought
                self.balance -= shares_bought * current_price
                
        elif action == 2: # Sell
            # 전량 매도 대신 절반 매도
            if self.shares_held > 0:
                shares_sold = int(self.shares_held * 0.5)
                # 최소 1주는 팔도록 보정
                if shares_sold == 0 and self.shares_held > 0: 
                    shares_sold = self.shares_held
                
                self.shares_held -= shares_sold
                self.balance += shares_sold * current_price
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # 보상 함수 개선 (Reward Shaping)
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        # 기본 보상: 자산 변화량
        reward = (self.net_worth - prev_net_worth)
        
        # 페널티 추가: 하락장에서 주식을 들고만 있으면 페널티 강화
        if action == 0 and self.net_worth < prev_net_worth:
            reward -= 10 # 손해보고 있는데 가만히 있으면 추가 벌점

        return self._get_obs(), reward, terminated, truncated, {'net_worth': self.net_worth}
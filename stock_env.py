import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf

class StockTradingEnv(gym.Env):
    def __init__(self, ticker='AAPL', start_date='2020-01-01', end_date='2023-01-01', initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        # 1. 데이터 로드 (yfinance 권장 사항 반영 [cite: 17])
        self.df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        self.df = self.df.reset_index()
        
        self.initial_balance = initial_balance
        
        # 2. Action Space 정의 (예: 0=Hold, 1=Buy, 2=Sell) [cite: 11]
        # 과제 초기 단계에서는 간단하게 1주씩 거래하거나 고정 수량을 거래하도록 설정 추천
        self.action_space = spaces.Discrete(3)
        
        # 3. Observation Space 정의 [cite: 10, 21]
        # 과제 예시처럼 Dict space 사용 [cite: 45]
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32), # [잔고, 보유주식수]
            "market": spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32) # [Open, High, Low, Close, Volume]
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        
        # 초기 상태 반환 [cite: 25]
        return self._get_obs(), {}

    def _get_obs(self):
        # 현재 단계의 시장 데이터 가져오기
        current_price_data = self.df.iloc[self.current_step]
        
        market_obs = np.array([
            current_price_data['Open'], current_price_data['High'], 
            current_price_data['Low'], current_price_data['Close'], 
            current_price_data['Volume']
        ], dtype=np.float32)
        
        agent_obs = np.array([self.balance, self.shares_held], dtype=np.float32)
        
        return {"agent": agent_obs, "market": market_obs}

    def step(self, action):
        # 현재 종가 가져오기
        current_price = self.df.iloc[self.current_step]['Close']
        
        # 행동 수행 (로직 구현 필요) [cite: 31, 32]
        # 예: 1주 매수/매도 로직
        if action == 1: # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2: # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
        
        # 다음 스텝으로 이동
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # 보상 계산 (포트폴리오 가치 변화) [cite: 14]
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        reward = self.net_worth - prev_net_worth # 수익이 나면 +보상
        
        obs = self._get_obs()
        
        return obs, reward, done, False, {'net_worth': self.net_worth}
